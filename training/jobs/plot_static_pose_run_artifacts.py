from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import torch
from torch.utils.data import DataLoader, Subset

from hlgam.ml.models.static_pose_transformer import StaticPoseTransformer, StaticPoseTransformerConfig
from hlgam.ml.schemas import LandmarkFeatureSpec
from training.common.config import load_yaml_config
from training.common.plotting import DEFAULT_VALIDATION_METRICS, plot_static_training_history, read_metric_history
from training.datasets.static_pose_dataset import StaticPoseDataset, static_pose_collate
from training.jobs.train_static_pose import move_batch_to_device, resolve_device, resolve_project_path


matplotlib.use("Agg")


DEFAULT_CHECKPOINT = "best.pt"


@dataclass(frozen=True)
class StaticPoseRunArtifactPaths:
    training_loss_plot: Path
    validation_metrics_plot: Path
    cumulative_time_plot: Path
    epoch_duration_plot: Path
    confusion_matrix_plot: Path
    confusion_matrix_csv: Path
    confusion_matrix_json: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "training_loss_plot": str(self.training_loss_plot),
            "validation_metrics_plot": str(self.validation_metrics_plot),
            "cumulative_time_plot": str(self.cumulative_time_plot),
            "epoch_duration_plot": str(self.epoch_duration_plot),
            "confusion_matrix_plot": str(self.confusion_matrix_plot),
            "confusion_matrix_csv": str(self.confusion_matrix_csv),
            "confusion_matrix_json": str(self.confusion_matrix_json),
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots for a completed static-pose training run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Static-pose run directory.")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint filename under checkpoints/ or an explicit checkpoint path.",
    )
    parser.add_argument("--split", default="test", help="Dataset split to evaluate for the confusion matrix.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string.")
    parser.add_argument("--threshold", type=float, help="Override the configured multilabel decision threshold.")
    parser.add_argument("--output-dir", type=Path, help="Output directory. Defaults to <run-dir>/metrics.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_paths = plot_static_pose_run_artifacts(
        args.run_dir,
        checkpoint=args.checkpoint,
        split=args.split,
        device_name=args.device,
        threshold=args.threshold,
        output_dir=args.output_dir,
    )
    for name, path in artifact_paths.to_dict().items():
        print(f"{name}={path}")
    return 0


def plot_static_pose_run_artifacts(
    run_dir: Path | str,
    *,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    split: str = "test",
    device_name: str = "auto",
    threshold: float | None = None,
    output_dir: Path | str | None = None,
) -> StaticPoseRunArtifactPaths:
    run_path = Path(run_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve() if output_dir is not None else run_path / "metrics"
    output_path.mkdir(parents=True, exist_ok=True)

    training_plots = plot_static_training_history(
        run_path,
        validation_metric_names=DEFAULT_VALIDATION_METRICS,
    )
    cumulative_time_plot, epoch_duration_plot = write_training_time_charts(run_path, output_path=output_path)
    confusion_paths = write_static_pose_confusion_matrix(
        run_path,
        checkpoint=checkpoint,
        split=split,
        device_name=device_name,
        threshold=threshold,
        output_path=output_path,
    )
    return StaticPoseRunArtifactPaths(
        training_loss_plot=training_plots.loss_plot,
        validation_metrics_plot=training_plots.validation_metrics_plot,
        cumulative_time_plot=cumulative_time_plot,
        epoch_duration_plot=epoch_duration_plot,
        confusion_matrix_plot=confusion_paths["plot"],
        confusion_matrix_csv=confusion_paths["csv"],
        confusion_matrix_json=confusion_paths["json"],
    )


def write_training_time_charts(run_dir: Path, *, output_path: Path) -> tuple[Path, Path]:
    validation_history = read_metric_history(run_dir / "metrics" / "validation_metrics.jsonl")
    train_history = read_metric_history(run_dir / "metrics" / "train_metrics.jsonl")
    history = validation_history or train_history
    if not history:
        raise ValueError(f"no timestamped metric history found for {run_dir}")

    summary = read_json_mapping(run_dir / "metrics" / "summary.json")
    elapsed_seconds = (
        float(summary.get("elapsed_seconds", 0.0))
        if isinstance(summary.get("elapsed_seconds"), int | float)
        else None
    )
    best_epoch = int(summary["best_epoch"]) if isinstance(summary.get("best_epoch"), int | float) else None
    epoch_times = epoch_time_points(history, elapsed_seconds=elapsed_seconds)

    cumulative_time_plot = output_path / "training_cumulative_time.png"
    write_cumulative_time_chart(cumulative_time_plot, epoch_times, best_epoch=best_epoch)
    epoch_duration_plot = output_path / "training_epoch_duration.png"
    write_epoch_duration_chart(epoch_duration_plot, epoch_times, best_epoch=best_epoch)
    return cumulative_time_plot, epoch_duration_plot


def epoch_time_points(
    history: Sequence[Mapping[str, Any]],
    *,
    elapsed_seconds: float | None,
) -> list[dict[str, float]]:
    rows = [
        row
        for row in history
        if isinstance(row.get("epoch"), int | float) and isinstance(row.get("timestamp"), str)
    ]
    if not rows:
        raise ValueError("metric history has no rows with epoch and timestamp")

    parsed = [(int(row["epoch"]), datetime.fromisoformat(str(row["timestamp"]))) for row in rows]
    parsed = sorted(parsed, key=lambda item: item[0])
    if elapsed_seconds is not None and elapsed_seconds > 0:
        estimated_start = parsed[-1][1] - timedelta(seconds=elapsed_seconds)
    else:
        estimated_start = parsed[0][1]

    points: list[dict[str, float]] = []
    previous_elapsed = 0.0
    for epoch, timestamp in parsed:
        cumulative_seconds = max(0.0, (timestamp - estimated_start).total_seconds())
        points.append(
            {
                "epoch": float(epoch),
                "cumulative_seconds": cumulative_seconds,
                "epoch_seconds": max(0.0, cumulative_seconds - previous_elapsed),
            }
        )
        previous_elapsed = cumulative_seconds
    return points


def write_cumulative_time_chart(path: Path, points: Sequence[Mapping[str, float]], *, best_epoch: int | None) -> None:
    from matplotlib import pyplot as plt

    epochs = [point["epoch"] for point in points]
    minutes = [point["cumulative_seconds"] / 60.0 for point in points]
    figure, axis = plt.subplots(figsize=(11, 6.5), dpi=130)
    axis.plot(epochs, minutes, marker="o", linewidth=2.2, markersize=4, color="#0891b2")
    annotate_best_epoch(axis, best_epoch)
    axis.set_title("Static Pose Training Cumulative Wall-Clock Time", fontsize=16, fontweight="bold", pad=14)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Elapsed minutes")
    style_time_axis(axis)
    figure.tight_layout()
    figure.savefig(path, format="png")
    plt.close(figure)


def write_epoch_duration_chart(path: Path, points: Sequence[Mapping[str, float]], *, best_epoch: int | None) -> None:
    from matplotlib import pyplot as plt

    epochs = [point["epoch"] for point in points]
    durations = [point["epoch_seconds"] for point in points]
    rolling = rolling_average(durations, window=5)
    figure, axis = plt.subplots(figsize=(11, 6.5), dpi=130)
    axis.bar(epochs, durations, color="#22d3ee", alpha=0.68, label="epoch duration")
    axis.plot(epochs, rolling, color="#f97316", linewidth=2.2, label="5-epoch rolling average")
    annotate_best_epoch(axis, best_epoch)
    axis.set_title("Static Pose Training Wall-Clock Time per Epoch", fontsize=16, fontweight="bold", pad=14)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Seconds")
    axis.legend(frameon=False)
    style_time_axis(axis)
    figure.tight_layout()
    figure.savefig(path, format="png")
    plt.close(figure)


def rolling_average(values: Sequence[float], *, window: int) -> list[float]:
    averaged: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        window_values = values[start:index + 1]
        averaged.append(sum(window_values) / max(1, len(window_values)))
    return averaged


def annotate_best_epoch(axis: Any, best_epoch: int | None) -> None:
    if best_epoch is None:
        return
    axis.axvline(best_epoch, color="#a855f7", linestyle="--", linewidth=1.7, alpha=0.85, label="best epoch")
    axis.text(best_epoch, 0.98, "best", transform=axis.get_xaxis_transform(), color="#a855f7", ha="right", va="top")


def style_time_axis(axis: Any) -> None:
    from matplotlib.ticker import MaxNLocator

    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def write_static_pose_confusion_matrix(
    run_dir: Path,
    *,
    checkpoint: str | Path,
    split: str,
    device_name: str,
    threshold: float | None,
    output_path: Path,
) -> dict[str, Path]:
    device = resolve_device(device_name)
    config = load_yaml_config(run_dir / "config.resolved.yaml")
    feature_spec = LandmarkFeatureSpec.from_mapping(read_json_mapping(run_dir / "input_feature_schema.json"))
    dataset = StaticPoseDataset(
        resolve_project_path(config["data"]["csv_path"]),
        label_schema_path=run_dir / "label_schema.json",
        feature_spec=feature_spec,
        limit_rows=config["data"].get("limit_rows"),
    )
    splits = read_json_mapping(run_dir / "metrics" / "splits.json")
    split_indices = splits.get(split)
    if not isinstance(split_indices, list) or not split_indices:
        raise ValueError(f"split {split!r} is not available in {run_dir / 'metrics' / 'splits.json'}")

    dataloader = DataLoader(
        Subset(dataset, [int(index) for index in split_indices]),
        batch_size=int(config["data"].get("batch_size", 64)),
        shuffle=False,
        num_workers=0,
        collate_fn=static_pose_collate,
    )
    model = load_static_pose_model_for_run(run_dir, config, dataset, feature_spec, checkpoint, device)
    decision_threshold = float(threshold if threshold is not None else config["training"].get("threshold", 0.5))
    targets, predictions = predict_static_pose_labels(model, dataloader, device, threshold=decision_threshold)
    records = confusion_records(targets, predictions, dataset.label_schema.labels)

    stem = f"{split}_multilabel_confusion_matrix"
    csv_path = output_path / f"{stem}.csv"
    json_path = output_path / f"{stem}.json"
    plot_path = output_path / f"{stem}.png"
    write_confusion_csv(csv_path, records)
    write_confusion_json(
        json_path,
        records,
        split=split,
        threshold=decision_threshold,
        checkpoint_path=resolve_checkpoint_path(run_dir, checkpoint),
    )
    write_confusion_heatmap(plot_path, records, split=split, threshold=decision_threshold)
    return {"plot": plot_path, "csv": csv_path, "json": json_path}


def load_static_pose_model_for_run(
    run_dir: Path,
    config: Mapping[str, Any],
    dataset: StaticPoseDataset,
    feature_spec: LandmarkFeatureSpec,
    checkpoint: str | Path,
    device: torch.device,
) -> StaticPoseTransformer:
    model_config = StaticPoseTransformerConfig(
        input_dim=feature_spec.per_landmark_dim,
        global_feature_dim=feature_spec.global_feature_dim,
        num_labels=dataset.num_labels,
        d_model=int(config["model"]["d_model"]),
        num_layers=int(config["model"]["num_layers"]),
        num_heads=int(config["model"]["num_heads"]),
        ffn_dim=int(config["model"]["ffn_dim"]),
        dropout=float(config["model"]["dropout"]),
        use_global_features=bool(config["model"].get("use_global_features", True)),
        auxiliary_heads=bool(config["model"].get("auxiliary_heads", True)),
    )
    model = StaticPoseTransformer(model_config).to(device)
    checkpoint_payload = torch.load(resolve_checkpoint_path(run_dir, checkpoint), map_location=device)
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model.eval()
    return model


def resolve_checkpoint_path(run_dir: Path, checkpoint: str | Path) -> Path:
    checkpoint_path = Path(checkpoint).expanduser()
    if checkpoint_path.is_absolute():
        return checkpoint_path
    candidate = run_dir / checkpoint_path
    if candidate.exists():
        return candidate
    return run_dir / "checkpoints" / checkpoint_path


def predict_static_pose_labels(
    model: StaticPoseTransformer,
    dataloader: DataLoader[dict[str, Any]],
    device: torch.device,
    *,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_parts: list[torch.Tensor] = []
    prediction_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in dataloader:
            moved_batch = move_batch_to_device(batch, device)
            outputs = model(moved_batch)
            probabilities = torch.sigmoid(outputs["logits"].detach().cpu())
            prediction_parts.append((probabilities >= threshold).to(torch.int64))
            target_parts.append(batch["labels"].detach().cpu().to(torch.int64))
    return torch.cat(target_parts, dim=0), torch.cat(prediction_parts, dim=0)


def confusion_records(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    label_names: Sequence[str],
) -> list[dict[str, float | int | str]]:
    records: list[dict[str, float | int | str]] = []
    for index, label in enumerate(label_names):
        target = targets[:, index]
        prediction = predictions[:, index]
        true_positive = int(((prediction == 1) & (target == 1)).sum().item())
        false_positive = int(((prediction == 1) & (target == 0)).sum().item())
        false_negative = int(((prediction == 0) & (target == 1)).sum().item())
        true_negative = int(((prediction == 0) & (target == 0)).sum().item())
        precision = safe_ratio(true_positive, true_positive + false_positive)
        recall = safe_ratio(true_positive, true_positive + false_negative)
        f1 = safe_ratio(2 * precision * recall, precision + recall)
        records.append(
            {
                "label": label,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "support": true_positive + false_negative,
                "predicted": true_positive + false_positive,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return records


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def write_confusion_csv(path: Path, records: Sequence[Mapping[str, float | int | str]]) -> None:
    columns = [
        "label",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "support",
        "predicted",
        "precision",
        "recall",
        "f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def write_confusion_json(
    path: Path,
    records: Sequence[Mapping[str, float | int | str]],
    *,
    split: str,
    threshold: float,
    checkpoint_path: Path,
) -> None:
    payload = {
        "split": split,
        "threshold": threshold,
        "checkpoint_path": str(checkpoint_path),
        "labels": list(records),
        "totals": {
            "true_positive": sum(int(record["true_positive"]) for record in records),
            "false_positive": sum(int(record["false_positive"]) for record in records),
            "false_negative": sum(int(record["false_negative"]) for record in records),
            "true_negative": sum(int(record["true_negative"]) for record in records),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_confusion_heatmap(
    path: Path,
    records: Sequence[Mapping[str, float | int | str]],
    *,
    split: str,
    threshold: float,
) -> None:
    from matplotlib import pyplot as plt

    columns = ("true_positive", "false_positive", "false_negative", "true_negative")
    labels = [str(record["label"]) for record in records]
    values = [[int(record[column]) for column in columns] for record in records]
    figure_height = max(8.0, len(labels) * 0.34)
    figure, axis = plt.subplots(figsize=(9.5, figure_height), dpi=130)
    image = axis.imshow(values, aspect="auto", cmap="mako" if has_colormap("mako") else "viridis")
    axis.set_title(
        f"Static Pose Multilabel Confusion Matrix ({split}, threshold={threshold:g})",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    axis.set_xticks(range(len(columns)), ["TP", "FP", "FN", "TN"])
    axis.set_yticks(range(len(labels)), labels)
    axis.tick_params(axis="y", labelsize=8)
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, str(value), ha="center", va="center", color="white", fontsize=7)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Count")
    figure.tight_layout()
    figure.savefig(path, format="png")
    plt.close(figure)


def has_colormap(name: str) -> bool:
    from matplotlib import colormaps

    return name in colormaps


def read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


if __name__ == "__main__":
    raise SystemExit(main())
