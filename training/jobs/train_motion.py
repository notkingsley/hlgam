from __future__ import annotations

import argparse
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from hlgam.ml.models.motion_transformer import MotionGestureTransformer, MotionGestureTransformerConfig
from hlgam.ml.schemas import DEFAULT_MOTION_GLOBAL_FEATURES, LandmarkFeatureSpec
from training.common.checkpointing import append_jsonl, save_checkpoint, write_json
from training.common.config import deep_merge, load_yaml_config, write_yaml_config
from training.common.distributed import (
    DistributedContext,
    distributed_summary,
    initialize_distributed_context,
    prepare_model_for_distributed,
    set_dataloader_epoch,
    unwrap_distributed_model,
)
from training.common.losses import MotionGestureLoss, MotionGestureLossConfig
from training.common.metrics import flatten_scalar_metrics, multilabel_classification_metrics
from training.common.plotting import DEFAULT_VALIDATION_METRICS, plot_motion_training_history
from training.common.seed import set_training_seed
from training.datasets.motion_window_dataset import (
    build_motion_window_dataloaders,
    compute_motion_pos_weight,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG: dict[str, Any] = {
    "run": {
        "name": "motion_transformer_v1",
        "seed": 1337,
        "output_root": "runs/motion",
    },
    "data": {
        "csv_path": "data/motion_sequences/motion_sequence_frames.csv",
        "label_schema_path": "training/label_schemas/motion_labels.json",
        "sequence_length": 32,
        "window_stride": 4,
        "min_interval_overlap": 0.25,
        "endpoint_positive": True,
        "batch_size": 32,
        "val_fraction": 0.15,
        "test_fraction": 0.15,
        "num_workers": 0,
        "limit_samples": None,
    },
    "features": {
        "coordinate_space": "normalized",
        "landmark_axes": ["x", "y", "z"],
        "global_feature_names": list(DEFAULT_MOTION_GLOBAL_FEATURES),
    },
    "model": {
        "spatial_d_model": 128,
        "spatial_layers": 2,
        "spatial_heads": 4,
        "spatial_ffn_dim": 512,
        "temporal_d_model": 256,
        "temporal_layers": 4,
        "temporal_heads": 8,
        "temporal_ffn_dim": 1024,
        "dropout": 0.1,
        "use_global_features": True,
        "causal": False,
    },
    "loss": {
        "primary_loss": "bce",
        "use_pos_weight": True,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "phase_weight": 0.5,
        "direction_weight": 0.1,
        "displacement_weight": 0.1,
        "velocity_weight": 0.1,
        "duration_weight": 0.05,
        "scale_weight": 0.05,
        "rotation_weight": 0.05,
        "draw_weight": 0.05,
    },
    "optimizer": {
        "lr": 0.0003,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
    },
    "scheduler": {
        "warmup_ratio": 0.05,
        "min_lr_ratio": 0.1,
    },
    "training": {
        "max_epochs": 150,
        "grad_clip_norm": 1.0,
        "amp": True,
        "early_stopping_patience": 35,
        "selection_metric": "macro_f1",
        "selection_mode": "max",
        "threshold": 0.5,
    },
    "checkpointing": {
        "save_best": True,
        "save_last": True,
    },
    "plots": {
        "enabled": True,
        "loss_filename": "training_validation_loss.png",
        "validation_metrics_filename": "validation_metrics.png",
        "validation_metric_names": [*DEFAULT_VALIDATION_METRICS, "phase_accuracy", "direction_accuracy"],
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the motion gesture transformer on captured sequence CSV rows.")
    parser.add_argument("--config", type=Path, help="YAML config file. Values override the built-in defaults.")
    parser.add_argument("--csv", type=Path, help="Motion sequence frame CSV path.")
    parser.add_argument("--label-schema", type=Path, help="Motion label schema JSON.")
    parser.add_argument("--output-root", type=Path, help="Root folder for run outputs.")
    parser.add_argument("--run-name", help="Human-readable run name suffix.")
    parser.add_argument("--epochs", type=int, help="Override max training epochs.")
    parser.add_argument("--batch-size", type=int, help="Override dataloader batch size.")
    parser.add_argument("--sequence-length", type=int, help="Override fixed motion window length.")
    parser.add_argument("--window-stride", type=int, help="Override endpoint stride between generated windows.")
    parser.add_argument("--limit-samples", type=int, help="Use only the first N motion samples for smoke tests.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string.")
    parser.add_argument("--dry-run", action="store_true", help="Build the pipeline and run one forward/loss pass only.")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation after training.")
    return parser.parse_args(argv)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def train_motion(
    config: Mapping[str, Any],
    *,
    device_name: str = "auto",
    dry_run: bool = False,
    distributed_context: DistributedContext | None = None,
) -> dict[str, Any]:
    resolved_config = deep_merge(DEFAULT_CONFIG, config)
    context = distributed_context or initialize_distributed_context(device_name)
    owns_context = distributed_context is None
    try:
        seed = int(resolved_config["run"]["seed"])
        set_training_seed(seed)
        device = context.device

        created_run_dir: Path | None = None
        if context.is_main_process:
            created_run_dir = create_run_dir(resolved_config)
            write_yaml_config(resolved_config, created_run_dir / "config.resolved.yaml")
        run_dir = Path(context.broadcast_object(str(created_run_dir)))

        feature_spec = LandmarkFeatureSpec.from_mapping(resolved_config["features"])
        dataloaders, dataset, split_indices = build_motion_window_dataloaders(
            resolve_project_path(resolved_config["data"]["csv_path"]),
            label_schema_path=resolve_project_path(resolved_config["data"]["label_schema_path"]),
            feature_spec=feature_spec,
            sequence_length=int(resolved_config["data"]["sequence_length"]),
            window_stride=int(resolved_config["data"]["window_stride"]),
            min_interval_overlap=float(resolved_config["data"]["min_interval_overlap"]),
            endpoint_positive=bool(resolved_config["data"].get("endpoint_positive", True)),
            batch_size=int(resolved_config["data"]["batch_size"]),
            val_fraction=float(resolved_config["data"]["val_fraction"]),
            test_fraction=float(resolved_config["data"]["test_fraction"]),
            seed=seed,
            num_workers=int(resolved_config["data"]["num_workers"]),
            limit_samples=resolved_config["data"].get("limit_samples"),
            pin_memory=device.type == "cuda",
            distributed_rank=context.rank,
            distributed_world_size=context.world_size,
        )
        if context.is_main_process:
            dataset.profile().write_json(run_dir / "metrics" / "dataset_profile.json")
            write_json(run_dir / "metrics" / "splits.json", split_indices)
            write_json(run_dir / "label_schema.json", dataset.schema_payload())
            write_json(run_dir / "input_feature_schema.json", feature_spec.to_dict())
            write_json(run_dir / "phase_labels.json", {"schema_version": 1, "labels": list(dataset.phase_labels)})

        model_config = build_model_config(resolved_config, dataset, feature_spec)
        base_model = MotionGestureTransformer(model_config).to(device)
        context.print(f"Trainable parameters: {count_parameters(base_model)}", flush=True)
        model = prepare_model_for_distributed(base_model, context)
        optimizer = build_optimizer(model, resolved_config)
        scheduler = build_scheduler(optimizer, resolved_config, max(1, len(dataloaders["train"])))
        pos_weight = None
        if bool(resolved_config["loss"].get("use_pos_weight", True)):
            pos_weight = compute_motion_pos_weight(dataloaders["train"].dataset).to(device)
        criterion = MotionGestureLoss(build_loss_config(resolved_config), pos_weight=pos_weight).to(device)

        training_config = resolved_config["training"]
        threshold = float(training_config["threshold"])
        max_epochs = int(training_config["max_epochs"])
        patience = int(training_config["early_stopping_patience"])
        selection_metric = str(training_config["selection_metric"])
        selection_mode = str(training_config["selection_mode"])
        use_amp = bool(training_config["amp"]) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        validation_loader = dataloaders.get("validation", dataloaders["train"])

        if dry_run:
            batch = next(iter(dataloaders["train"]))
            batch = move_batch_to_device(batch, device)
            with torch.inference_mode():
                outputs = model(batch)
                loss, loss_parts = criterion(outputs, batch)
            batch_size = batch["labels"].shape[0]
            loss_value = context.reduce_weighted_average(float(loss.detach().cpu()) * batch_size, batch_size)
            summary = None
            if context.is_main_process:
                summary = {
                    "run_dir": str(run_dir),
                    "dry_run": True,
                    "loss": loss_value,
                    "loss_parts": loss_parts,
                    "output_shapes": {key: list(value.shape) for key, value in outputs.items()},
                    "distributed": distributed_summary(context),
                }
                write_json(run_dir / "metrics" / "dry_run_summary.json", summary)
            return context.broadcast_object(summary)

        best_metric: float | None = None
        best_epoch = 0
        global_step = 0
        epochs_without_improvement = 0

        start_time = time.time()
        start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        context.print(f"Starting training with {max_epochs=} at {start_time_str} on {device=}...", flush=True)

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()
            train_metrics, global_step = train_one_epoch(
                model,
                dataloaders["train"],
                criterion,
                optimizer,
                scheduler,
                scaler,
                device,
                context,
                epoch=epoch,
                global_step=global_step,
                grad_clip_norm=float(training_config["grad_clip_norm"]),
                use_amp=use_amp,
            )
            should_stop = False
            if context.is_main_process:
                val_metrics = evaluate_motion(
                    unwrap_distributed_model(model),
                    validation_loader,
                    criterion,
                    device,
                    label_names=dataset.label_schema.labels,
                    threshold=threshold,
                    split="validation",
                )
                append_epoch_metrics(run_dir, "train", epoch, global_step, train_metrics)
                append_epoch_metrics(run_dir, "validation", epoch, global_step, val_metrics)

                current_metric = float(
                    val_metrics.get(selection_metric, math.inf if selection_mode == "min" else -math.inf)
                )
                improved = is_improved(current_metric, best_metric, selection_mode)
                if improved:
                    best_metric = current_metric
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    if resolved_config["checkpointing"].get("save_best", True):
                        save_checkpoint(
                            run_dir / "checkpoints" / "best.pt",
                            model=unwrap_distributed_model(model),
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            global_step=global_step,
                            best_metric=best_metric,
                            config=resolved_config,
                            metrics=val_metrics,
                        )
                else:
                    epochs_without_improvement += 1

                if resolved_config["checkpointing"].get("save_last", True):
                    save_checkpoint(
                        run_dir / "checkpoints" / "last.pt",
                        model=unwrap_distributed_model(model),
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        best_metric=best_metric,
                        config=resolved_config,
                        metrics=val_metrics,
                    )

                show_metrics = (
                    "loss",
                    "phase_accuracy",
                    "direction_accuracy",
                    "exact_match",
                    "macro_precision",
                    "macro_recall",
                )
                print(
                    f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
                    f"{format_metric_values('val', val_metrics, show_metrics)} "
                    f"val_{selection_metric}={current_metric:.4f} "
                    f"elapsed={time.time() - epoch_start:.1f}s",
                    flush=True,
                )
                if patience > 0 and epochs_without_improvement >= patience:
                    print(f"early stopping after {epochs_without_improvement=}", flush=True)
                    should_stop = True
            if context.broadcast_object(should_stop):
                break

        summary = None
        if context.is_main_process:
            test_metrics = None
            if "test" in dataloaders:
                test_metrics = evaluate_motion(
                    unwrap_distributed_model(model),
                    dataloaders["test"],
                    criterion,
                    device,
                    label_names=dataset.label_schema.labels,
                    threshold=threshold,
                    split="test",
                )
                write_json(run_dir / "metrics" / "test_metrics.json", test_metrics)
                print(
                    "test evaluation "
                    + format_metric_values(
                        "test",
                        test_metrics,
                        ("loss", "macro_precision", "macro_recall", "exact_match", "macro_f1", "phase_accuracy"),
                    ),
                    flush=True,
                )
            else:
                print("test evaluation skipped: no test split configured", flush=True)

            plot_outputs = None
            plots_config = resolved_config.get("plots", {})
            if isinstance(plots_config, Mapping) and bool(plots_config.get("enabled", True)):
                plot_paths = plot_motion_training_history(
                    run_dir,
                    validation_metric_names=tuple(
                        plots_config.get("validation_metric_names", DEFAULT_VALIDATION_METRICS)
                    ),
                    loss_filename=str(plots_config.get("loss_filename", "training_validation_loss.png")),
                    validation_metrics_filename=str(
                        plots_config.get("validation_metrics_filename", "validation_metrics.png")
                    ),
                )
                plot_outputs = plot_paths.to_dict()

            summary = {
                "run_dir": str(run_dir),
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "selection_metric": selection_metric,
                "selection_mode": selection_mode,
                "elapsed_seconds": time.time() - start_time,
                "model_config": model_config.to_dict(),
                "test_metrics": test_metrics,
                "plots": plot_outputs,
                "distributed": distributed_summary(context),
            }
            write_json(run_dir / "metrics" / "summary.json", summary)
        return context.broadcast_object(summary)
    finally:
        if owns_context:
            context.close()


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader[dict[str, Any]],
    criterion: MotionGestureLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    context: DistributedContext,
    *,
    epoch: int,
    global_step: int,
    grad_clip_norm: float,
    use_amp: bool,
) -> tuple[dict[str, float], int]:
    model.train()
    set_dataloader_epoch(dataloader, epoch)
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch)
            loss, _ = criterion(outputs, batch)
        batch_size = batch["labels"].shape[0]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        global_step += 1
    average_loss = context.reduce_weighted_average(total_loss, total_samples)
    return {"loss": average_loss, "epoch": float(epoch)}, global_step


def evaluate_motion(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader[dict[str, Any]],
    criterion: MotionGestureLoss,
    device: torch.device,
    *,
    label_names: Sequence[str],
    threshold: float,
    split: str,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    logits_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    phase_correct = 0
    phase_total = 0
    direction_correct = 0
    direction_total = 0
    regression_errors: dict[str, list[float]] = {}
    with torch.inference_mode():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            loss, _ = criterion(outputs, batch)
            batch_size = batch["labels"].shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size
            logits_parts.append(outputs["action_logits"].detach().cpu())
            target_parts.append(batch["labels"].detach().cpu())
            phase_batch_correct, phase_batch_total = phase_accuracy_counts(outputs, batch)
            phase_correct += phase_batch_correct
            phase_total += phase_batch_total
            direction_batch_correct, direction_batch_total = direction_accuracy_counts(outputs, batch)
            direction_correct += direction_batch_correct
            direction_total += direction_batch_total
            collect_regression_errors(outputs, batch, regression_errors)
    logits = torch.cat(logits_parts, dim=0)
    targets = torch.cat(target_parts, dim=0)
    metrics = multilabel_classification_metrics(logits, targets, label_names=label_names, threshold=threshold)
    metrics["loss"] = total_loss / max(1, total_samples)
    metrics["split"] = split
    metrics["phase_accuracy"] = phase_correct / phase_total if phase_total else 0.0
    metrics["direction_accuracy"] = direction_correct / direction_total if direction_total else 0.0
    for metric_name, values in regression_errors.items():
        metrics[f"{metric_name}_mae"] = sum(values) / max(1, len(values))
    return metrics


def phase_accuracy_counts(outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any]) -> tuple[int, int]:
    logits = outputs["phase_logits"]
    targets = batch["phase_targets"].to(logits.device)
    mask = batch["phase_mask"].to(logits.device).bool()
    if not torch.any(mask):
        return 0, 0
    predictions = logits.argmax(dim=-1)
    correct = (predictions[mask] == targets[mask]).sum().item()
    total = mask.sum().item()
    return int(correct), int(total)


def direction_accuracy_counts(outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any]) -> tuple[int, int]:
    aux = batch.get("aux", {})
    if "direction_logits" not in outputs or "direction" not in aux:
        return 0, 0
    logits = outputs["direction_logits"]
    targets = aux["direction"].to(logits.device)
    mask = aux.get("direction_mask", torch.zeros_like(targets, dtype=torch.bool)).to(logits.device).bool()
    if not torch.any(mask):
        return 0, 0
    predictions = logits.argmax(dim=-1)
    correct = (predictions[mask] == targets[mask]).sum().item()
    total = mask.sum().item()
    return int(correct), int(total)


def collect_regression_errors(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    errors: dict[str, list[float]],
) -> None:
    aux = batch.get("aux", {})
    for target_name, mask_name in (
        ("displacement", "displacement_mask"),
        ("velocity", "velocity_mask"),
        ("duration_ms", "duration_mask"),
        ("scale_delta", "scale_mask"),
        ("rotation_degrees", "rotation_mask"),
        ("draw_point", "draw_mask"),
    ):
        if target_name not in outputs or target_name not in aux or mask_name not in aux:
            continue
        predictions = outputs[target_name]
        targets = aux[target_name].to(predictions.device)
        mask = aux[mask_name].to(predictions.device).bool()
        if not torch.any(mask):
            continue
        mae = torch.abs(predictions[mask] - targets[mask]).detach().cpu()
        errors.setdefault(target_name, []).extend(mae.reshape(mae.shape[0], -1).mean(dim=1).tolist())


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, Mapping):
            moved[key] = {nested_key: nested_value.to(device) for nested_key, nested_value in value.items()}
        else:
            moved[key] = value
    return moved


def build_model_config(
    config: Mapping[str, Any],
    dataset: Any,
    feature_spec: LandmarkFeatureSpec,
) -> MotionGestureTransformerConfig:
    model_config = config["model"]
    sequence_length = int(config["data"]["sequence_length"])
    return MotionGestureTransformerConfig(
        input_dim=feature_spec.per_landmark_dim,
        global_feature_dim=feature_spec.global_feature_dim,
        num_labels=dataset.num_labels,
        num_phases=dataset.num_phases,
        spatial_d_model=int(model_config["spatial_d_model"]),
        spatial_layers=int(model_config["spatial_layers"]),
        spatial_heads=int(model_config["spatial_heads"]),
        spatial_ffn_dim=int(model_config["spatial_ffn_dim"]),
        temporal_d_model=int(model_config["temporal_d_model"]),
        temporal_layers=int(model_config["temporal_layers"]),
        temporal_heads=int(model_config["temporal_heads"]),
        temporal_ffn_dim=int(model_config["temporal_ffn_dim"]),
        max_sequence_length=max(sequence_length, int(model_config.get("max_sequence_length", sequence_length))),
        dropout=float(model_config["dropout"]),
        use_global_features=bool(model_config["use_global_features"]),
        causal=bool(model_config.get("causal", False)),
    )


def build_loss_config(config: Mapping[str, Any]) -> MotionGestureLossConfig:
    loss_config = config["loss"]
    return MotionGestureLossConfig(
        primary_loss=str(loss_config["primary_loss"]),
        focal_alpha=float(loss_config["focal_alpha"]),
        focal_gamma=float(loss_config["focal_gamma"]),
        phase_weight=float(loss_config["phase_weight"]),
        direction_weight=float(loss_config["direction_weight"]),
        displacement_weight=float(loss_config["displacement_weight"]),
        velocity_weight=float(loss_config["velocity_weight"]),
        duration_weight=float(loss_config["duration_weight"]),
        scale_weight=float(loss_config["scale_weight"]),
        rotation_weight=float(loss_config["rotation_weight"]),
        draw_weight=float(loss_config["draw_weight"]),
    )


def build_optimizer(model: nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    optimizer_config = config["optimizer"]
    betas = optimizer_config.get("betas", [0.9, 0.999])
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        betas=(float(betas[0]), float(betas[1])),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    max_epochs = int(config["training"]["max_epochs"])
    total_steps = max(1, max_epochs * steps_per_epoch)
    warmup_steps = int(total_steps * float(config["scheduler"]["warmup_ratio"]))
    min_lr_ratio = float(config["scheduler"]["min_lr_ratio"])

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def append_epoch_metrics(run_dir: Path, split: str, epoch: int, global_step: int, metrics: Mapping[str, Any]) -> None:
    append_jsonl(
        run_dir / "metrics" / f"{split}_metrics.jsonl",
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "split": split,
            "epoch": epoch,
            "global_step": global_step,
            "metrics": flatten_scalar_metrics(metrics),
        },
    )


def format_metric_values(prefix: str, metrics: Mapping[str, Any], metric_names: Sequence[str]) -> str:
    return " ".join(f"{prefix}_{metric_name}={metric_value(metrics, metric_name):.4f}" for metric_name in metric_names)


def metric_value(metrics: Mapping[str, Any], metric_name: str) -> float:
    value = metrics.get(metric_name, float("nan"))
    if isinstance(value, int | float):
        return float(value)
    return float("nan")


def create_run_dir(config: Mapping[str, Any]) -> Path:
    output_root = resolve_project_path(config["run"]["output_root"])
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_name = str(config["run"].get("name") or "motion_transformer")
    run_dir = output_root / f"{timestamp}_{slugify(run_name)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def is_improved(value: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    if mode == "min":
        return value < best
    if mode == "max":
        return value > best
    raise ValueError("selection_mode must be max or min")


def slugify(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.lower()).strip("_")


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml_config(args.config)
    overrides: dict[str, Any] = {}
    if args.csv is not None:
        overrides.setdefault("data", {})["csv_path"] = str(args.csv)
    if args.label_schema is not None:
        overrides.setdefault("data", {})["label_schema_path"] = str(args.label_schema)
    if args.output_root is not None:
        overrides.setdefault("run", {})["output_root"] = str(args.output_root)
    if args.run_name is not None:
        overrides.setdefault("run", {})["name"] = args.run_name
    if args.epochs is not None:
        overrides.setdefault("training", {})["max_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides.setdefault("data", {})["batch_size"] = args.batch_size
    if args.sequence_length is not None:
        overrides.setdefault("data", {})["sequence_length"] = args.sequence_length
    if args.window_stride is not None:
        overrides.setdefault("data", {})["window_stride"] = args.window_stride
    if args.limit_samples is not None:
        overrides.setdefault("data", {})["limit_samples"] = args.limit_samples
    if args.no_plots:
        overrides.setdefault("plots", {})["enabled"] = False
    return deep_merge(config, overrides)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    summary = train_motion(config, device_name=args.device, dry_run=args.dry_run)
    print(f"run_dir={summary['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
