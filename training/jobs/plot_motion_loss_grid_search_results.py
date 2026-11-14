from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.common.config import load_yaml_config


IMPORTANT_MOTION_METRICS = (
    "macro_recall",
    "macro_precision",
    "macro_f1",
    "exact_match",
)

METRIC_LABELS = {
    "macro_recall": "Macro Recall",
    "macro_precision": "Macro Precision",
    "macro_f1": "Macro F1",
    "exact_match": "Exact Match",
}

METRIC_COLORS = {
    "macro_recall": "#2563eb",
    "macro_precision": "#f97316",
    "macro_f1": "#16a34a",
    "exact_match": "#9333ea",
}

MISSING_VALUE = "(missing)"


@dataclass(frozen=True)
class MotionLossGridTrial:
    trial_id: str
    run_dir: Path | None
    loss_config: dict[str, Any]
    metrics: dict[str, float]
    best_epoch: int | None = None
    best_metric: float | None = None
    score: float | None = None

    def to_record(self, parameter_names: Sequence[str], metric_names: Sequence[str]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "trial_id": self.trial_id,
            "run_dir": str(self.run_dir) if self.run_dir is not None else "",
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "score": self.score,
        }
        for parameter_name in parameter_names:
            record[f"loss.{parameter_name}"] = self.loss_config.get(parameter_name, MISSING_VALUE)
        for metric_name in metric_names:
            record[metric_name] = self.metrics.get(metric_name)
        return record


@dataclass(frozen=True)
class MotionLossGridPlotPaths:
    plots_dir: Path
    performance_plot: Path
    ranking_csv: Path
    ranking_json: Path
    parameter_metric_plots: dict[str, Path]

    def to_dict(self) -> dict[str, object]:
        return {
            "plots_dir": str(self.plots_dir),
            "performance_plot": str(self.performance_plot),
            "ranking_csv": str(self.ranking_csv),
            "ranking_json": str(self.ranking_json),
            "parameter_metric_plots": {
                parameter_name: str(plot_path)
                for parameter_name, plot_path in self.parameter_metric_plots.items()
            },
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot motion loss-grid performance across macro recall, precision, F1, and exact match."
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help=(
            "Path to a motion loss-grid run directory, its trials/ directory, "
            "or a grid_search_results.json file."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for generated plots and ranking exports. Defaults under the results path.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(IMPORTANT_MOTION_METRICS),
        help="Metrics to plot. Defaults to macro_recall macro_precision macro_f1 exact_match.",
    )
    parser.add_argument("--sort-by", default="macro_f1", help="Metric used to rank the overview plot and CSV.")
    parser.add_argument("--ascending", action="store_true", help="Sort smaller metric values first.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of ranked trials to show in the overview plot. Use 0 to show all trials.",
    )
    parser.add_argument("--format", default="png", help="Matplotlib output format, such as png, svg, or pdf.")
    parser.add_argument("--dpi", type=int, default=140, help="Output DPI for raster formats.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plot_paths = plot_motion_loss_grid_search_results(
        args.results,
        output_dir=args.output_dir,
        metric_names=args.metrics,
        sort_by=args.sort_by,
        ascending=args.ascending,
        top_n=args.top_n,
        image_format=args.format,
        dpi=args.dpi,
    )
    for name, value in plot_paths.to_dict().items():
        if isinstance(value, dict):
            for parameter_name, plot_path in value.items():
                print(f"{name}.{parameter_name}={plot_path}")
        else:
            print(f"{name}={value}")
    return 0


def plot_motion_loss_grid_search_results(
    results_path: Path | str,
    *,
    output_dir: Path | str | None = None,
    metric_names: Sequence[str] | None = None,
    sort_by: str = "macro_f1",
    ascending: bool = False,
    top_n: int = 30,
    image_format: str = "png",
    dpi: int = 140,
) -> MotionLossGridPlotPaths:
    trials = collect_motion_loss_grid_trials(results_path)
    metrics = tuple(normalize_metric_name(metric_name) for metric_name in (metric_names or IMPORTANT_MOTION_METRICS))
    missing_metrics = sorted(
        {
            metric_name
            for metric_name in metrics
            if not any(metric_name in trial.metrics for trial in trials)
        }
    )
    if missing_metrics:
        raise ValueError(f"metrics not found in completed trials: {', '.join(missing_metrics)}")

    sort_metric = normalize_metric_name(sort_by)
    sorted_trials = sort_trials(trials, sort_metric, ascending=ascending)
    parameter_names = varying_loss_parameters(sorted_trials)

    plots_dir = resolve_output_dir(results_path, output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    normalized_format = image_format.lower().lstrip(".") or "png"

    performance_plot = plots_dir / f"motion_loss_grid_performance.{normalized_format}"
    write_trial_performance_plot(
        performance_plot,
        trials=sorted_trials,
        metric_names=metrics,
        sort_by=sort_metric,
        top_n=top_n,
        image_format=normalized_format,
        dpi=dpi,
    )

    parameter_metric_plots: dict[str, Path] = {}
    for parameter_name in parameter_names:
        plot_path = plots_dir / f"motion_loss_grid_{slugify(parameter_name)}.{normalized_format}"
        write_parameter_metric_plot(
            plot_path,
            trials=sorted_trials,
            parameter_name=parameter_name,
            metric_names=metrics,
            sort_by=sort_metric,
            image_format=normalized_format,
            dpi=dpi,
        )
        parameter_metric_plots[parameter_name] = plot_path

    ranking_csv = plots_dir / "motion_loss_grid_ranked_trials.csv"
    ranking_json = plots_dir / "motion_loss_grid_ranked_trials.json"
    write_ranked_trial_outputs(
        ranking_csv,
        ranking_json,
        trials=sorted_trials,
        parameter_names=all_loss_parameters(sorted_trials),
        metric_names=metrics,
    )
    return MotionLossGridPlotPaths(
        plots_dir=plots_dir,
        performance_plot=performance_plot,
        ranking_csv=ranking_csv,
        ranking_json=ranking_json,
        parameter_metric_plots=parameter_metric_plots,
    )


def collect_motion_loss_grid_trials(results_path: Path | str) -> list[MotionLossGridTrial]:
    results = Path(results_path).expanduser()
    if results.is_file():
        return trials_from_grid_search_results(results)
    if not results.exists():
        raise FileNotFoundError(f"results path not found: {results}")

    aggregate_results = results / "grid_search_results.json"
    if aggregate_results.exists():
        return trials_from_grid_search_results(aggregate_results)

    trial_root = results / "trials" if (results / "trials").is_dir() else results
    trials = trials_from_trial_directories(trial_root)
    if not trials:
        raise ValueError(f"no trial metrics found under {results}")
    return trials


def trials_from_grid_search_results(results_file: Path) -> list[MotionLossGridTrial]:
    payload = read_json_mapping(results_file)
    raw_trials = payload.get("trials")
    if not isinstance(raw_trials, list):
        summary = payload.get("summary")
        raw_trials = summary.get("trials") if isinstance(summary, Mapping) else None
    if not isinstance(raw_trials, list):
        raise ValueError(f"grid-search results must contain a trials list: {results_file}")

    trials: list[MotionLossGridTrial] = []
    for raw_trial in raw_trials:
        if not isinstance(raw_trial, Mapping) or raw_trial.get("status") != "completed":
            continue
        metrics = finite_scalar_mapping(raw_trial.get("test_metrics"))
        run_dir = existing_run_dir(raw_trial.get("run_dir"), results_file.parent)
        if not metrics and run_dir is not None:
            metrics = finite_scalar_mapping(read_optional_json_mapping(run_dir / "metrics" / "test_metrics.json"))
        trials.append(
            MotionLossGridTrial(
                trial_id=str(raw_trial.get("trial_id") or infer_trial_id(run_dir.name if run_dir else "trial")),
                run_dir=run_dir,
                loss_config=dict(raw_trial.get("loss_config") or {}),
                metrics=metrics,
                best_epoch=optional_int(raw_trial.get("best_epoch")),
                best_metric=optional_float(raw_trial.get("best_metric")),
                score=optional_float(raw_trial.get("score")),
            )
        )
    if not trials:
        raise ValueError(f"no completed trials found in {results_file}")
    return trials


def trials_from_trial_directories(trial_root: Path) -> list[MotionLossGridTrial]:
    metric_paths = sorted(trial_root.glob("*/metrics/test_metrics.json"))
    if not metric_paths:
        metric_paths = sorted(trial_root.rglob("metrics/test_metrics.json"))

    trials: list[MotionLossGridTrial] = []
    for metrics_path in metric_paths:
        run_dir = metrics_path.parent.parent
        metrics = finite_scalar_mapping(read_json_mapping(metrics_path))
        config = read_trial_config(run_dir)
        summary = read_optional_json_mapping(run_dir / "metrics" / "summary.json")
        grid_search = config.get("grid_search") if isinstance(config.get("grid_search"), Mapping) else {}
        loss_config = grid_search.get("loss_config") if isinstance(grid_search, Mapping) else None
        if not isinstance(loss_config, Mapping):
            loss_config = config.get("loss") if isinstance(config.get("loss"), Mapping) else {}
        trial_id = grid_search.get("trial_id") if isinstance(grid_search, Mapping) else None
        trials.append(
            MotionLossGridTrial(
                trial_id=str(trial_id or infer_trial_id(run_dir.name)),
                run_dir=run_dir,
                loss_config=dict(loss_config),
                metrics=metrics,
                best_epoch=optional_int(summary.get("best_epoch")),
                best_metric=optional_float(summary.get("best_metric")),
                score=optional_float(summary.get("best_metric")),
            )
        )
    return sorted(trials, key=trial_sort_key)


def read_trial_config(run_dir: Path) -> dict[str, Any]:
    for config_name in ("config.resolved.yaml", "config.yaml"):
        config_path = run_dir / config_name
        if config_path.exists():
            return load_yaml_config(config_path)
    return {}


def read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload must be an object: {path}")
    return dict(payload)


def read_optional_json_mapping(path: Path) -> dict[str, Any]:
    return read_json_mapping(path) if path.exists() else {}


def finite_scalar_mapping(payload: Any) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    values: dict[str, float] = {}
    for key, value in payload.items():
        numeric_value = optional_float(value)
        if numeric_value is not None:
            values[str(key)] = numeric_value
    return values


def existing_run_dir(value: Any, fallback_root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if candidate.exists():
        return candidate
    fallback_candidate = fallback_root / candidate.name
    return fallback_candidate if fallback_candidate.exists() else None


def optional_float(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    numeric_value = float(value)
    return numeric_value if math.isfinite(numeric_value) else None


def optional_int(value: Any) -> int | None:
    numeric_value = optional_float(value)
    return int(numeric_value) if numeric_value is not None else None


def normalize_metric_name(metric_name: str) -> str:
    return metric_name.removeprefix("test.")


def sort_trials(
    trials: Sequence[MotionLossGridTrial],
    metric_name: str,
    *,
    ascending: bool,
) -> list[MotionLossGridTrial]:
    def metric_sort_key(trial: MotionLossGridTrial) -> tuple[int, float, tuple[int, str]]:
        metric_value = trial.metrics.get(metric_name)
        if metric_value is None:
            return (1, 0.0, trial_sort_key(trial))
        return (0, metric_value, trial_sort_key(trial))

    return sorted(trials, key=metric_sort_key, reverse=not ascending)


def trial_sort_key(trial: MotionLossGridTrial) -> tuple[int, str]:
    match = re.search(r"(\d+)", trial.trial_id)
    if match:
        return (int(match.group(1)), trial.trial_id)
    return (10**9, trial.trial_id)


def infer_trial_id(name: str) -> str:
    match = re.search(r"trial[_-](\d+)", name)
    if not match:
        return name
    return f"trial_{int(match.group(1)):03d}"


def all_loss_parameters(trials: Sequence[MotionLossGridTrial]) -> list[str]:
    return sorted({key for trial in trials for key in trial.loss_config})


def varying_loss_parameters(trials: Sequence[MotionLossGridTrial]) -> list[str]:
    varying: list[str] = []
    for parameter_name in all_loss_parameters(trials):
        values = {canonical_value(trial.loss_config.get(parameter_name, MISSING_VALUE)) for trial in trials}
        if len(values) > 1:
            varying.append(parameter_name)
    return varying


def canonical_value(value: Any) -> Any:
    if isinstance(value, bool | int | float | str):
        return value
    if value is None:
        return "None"
    return str(value)


def sorted_parameter_values(values: set[Any]) -> list[Any]:
    return sorted(values, key=parameter_sort_key)


def parameter_sort_key(value: Any) -> tuple[int, float | str]:
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, int | float):
        return (2, float(value))
    return (3, str(value))


def write_trial_performance_plot(
    path: Path,
    *,
    trials: Sequence[MotionLossGridTrial],
    metric_names: Sequence[str],
    sort_by: str,
    top_n: int,
    image_format: str,
    dpi: int,
) -> None:
    from matplotlib import pyplot as plt

    display_trials = list(trials[:top_n]) if top_n > 0 else list(trials)
    if not display_trials:
        raise ValueError("no trials available for plotting")

    path.parent.mkdir(parents=True, exist_ok=True)
    positions = list(range(len(display_trials)))
    bar_width = min(0.18, 0.82 / max(1, len(metric_names)))
    figure_width = max(12.0, min(30.0, len(display_trials) * 0.7 + 4.0))
    figure, axis = plt.subplots(figsize=(figure_width, 7.0), dpi=dpi)

    for metric_index, metric_name in enumerate(metric_names):
        offset = (metric_index - (len(metric_names) - 1) / 2.0) * bar_width
        values = [trial.metrics.get(metric_name, float("nan")) for trial in display_trials]
        axis.bar(
            [position + offset for position in positions],
            values,
            width=bar_width,
            label=metric_label(metric_name),
            color=METRIC_COLORS.get(metric_name),
            alpha=0.88,
        )

    axis.set_title("Motion Loss Grid Search Performance", fontsize=17, fontweight="bold", pad=16)
    axis.set_xlabel(f"Trials ranked by {metric_label(sort_by)}")
    axis.set_ylabel("Metric value")
    axis.set_xticks(positions, [trial.trial_id for trial in display_trials], rotation=45, ha="right")
    axis.set_ylim(0.0, 1.03)
    axis.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    figure.tight_layout()
    figure.savefig(path, format=image_format)
    plt.close(figure)


def write_parameter_metric_plot(
    path: Path,
    *,
    trials: Sequence[MotionLossGridTrial],
    parameter_name: str,
    metric_names: Sequence[str],
    sort_by: str,
    image_format: str,
    dpi: int,
) -> None:
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    path.parent.mkdir(parents=True, exist_ok=True)
    parameter_values = sorted_parameter_values(
        {canonical_value(trial.loss_config.get(parameter_name, MISSING_VALUE)) for trial in trials}
    )
    value_positions = {value: index for index, value in enumerate(parameter_values)}
    numeric_axis = all(isinstance(value, int | float) and not isinstance(value, bool) for value in parameter_values)
    columns = 2
    rows = math.ceil(len(metric_names) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 6.0, rows * 4.0), dpi=dpi)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    best_trial_id = trials[0].trial_id if trials else None

    for axis, metric_name in zip(axes_list, metric_names, strict=False):
        metric_values: list[float] = []
        for trial_index, trial in enumerate(trials):
            metric_value = trial.metrics.get(metric_name)
            if metric_value is None:
                continue
            parameter_value = canonical_value(trial.loss_config.get(parameter_name, MISSING_VALUE))
            axis.scatter(
                parameter_x_value(parameter_value, value_positions, numeric_axis, trial_index, parameter_values),
                metric_value,
                s=130 if trial.trial_id == best_trial_id else 50,
                marker="*" if trial.trial_id == best_trial_id else "o",
                color=METRIC_COLORS.get(metric_name, "#0f172a"),
                alpha=0.9 if trial.trial_id == best_trial_id else 0.72,
                edgecolors="black" if trial.trial_id == best_trial_id else "none",
                linewidths=0.8 if trial.trial_id == best_trial_id else 0.0,
            )
            metric_values.append(metric_value)

        axis.set_title(metric_label(metric_name), fontsize=12, fontweight="bold")
        axis.set_xlabel(parameter_name)
        axis.set_ylabel("Metric value")
        axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.32)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if numeric_axis:
            axis.set_xticks([float(value) for value in parameter_values])
        else:
            axis.set_xticks(
                list(value_positions.values()),
                [format_parameter_value(value) for value in parameter_values],
                rotation=35,
                ha="right",
            )
        if metric_values and min(metric_values) >= 0.0 and max(metric_values) <= 1.0:
            axis.set_ylim(0.0, 1.03)

    for axis in axes_list[len(metric_names):]:
        axis.set_visible(False)

    figure.suptitle(
        f"Motion Loss Grid Search Metrics by {parameter_name}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    figure.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#64748b", markersize=7, label="trial"),
            Line2D(
                [0],
                [0],
                marker="*",
                color="black",
                markerfacecolor="#64748b",
                markersize=12,
                label=f"top {metric_label(sort_by)}",
            ),
        ],
        loc="upper right",
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    figure.savefig(path, format=image_format)
    plt.close(figure)


def parameter_x_value(
    value: Any,
    value_positions: Mapping[Any, int],
    numeric_axis: bool,
    trial_index: int,
    parameter_values: Sequence[Any],
) -> float:
    if numeric_axis:
        base = float(value)
        numeric_values = [float(parameter_value) for parameter_value in parameter_values]
        span = max(numeric_values) - min(numeric_values)
        jitter_scale = span * 0.012 if span > 0 else 0.02
    else:
        base = float(value_positions[value])
        jitter_scale = 0.12
    return base + deterministic_jitter(trial_index, jitter_scale)


def deterministic_jitter(index: int, scale: float) -> float:
    return (((index * 17) % 13) - 6) / 6 * scale


def write_ranked_trial_outputs(
    csv_path: Path,
    json_path: Path,
    *,
    trials: Sequence[MotionLossGridTrial],
    parameter_names: Sequence[str],
    metric_names: Sequence[str],
) -> None:
    records = [trial.to_record(parameter_names, metric_names) for trial in trials]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(records[0]) if records else []
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({key: csv_value(value) for key, value in record.items()})

    json_path.write_text(json.dumps(records, indent=2, sort_keys=True, default=str), encoding="utf-8")


def csv_value(value: Any) -> Any:
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True)
    return value


def resolve_output_dir(results_path: Path | str, output_dir: Path | str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser()
    results = Path(results_path).expanduser()
    base_dir = results.parent if results.is_file() else results
    return base_dir / "plots" / "motion_loss_grid_performance"


def metric_label(metric_name: str) -> str:
    return METRIC_LABELS.get(metric_name, metric_name.replace("_", " ").title())


def format_parameter_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def slugify(value: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "parameter"


if __name__ == "__main__":
    raise SystemExit(main())
