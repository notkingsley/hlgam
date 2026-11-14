from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")


DEFAULT_VALIDATION_METRICS = (
    "macro_f1",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "micro_precision",
    "micro_recall",
    "exact_match",
    "hamming_loss",
)

DEFAULT_GRID_SEARCH_METRICS = (
    "score",
    "best_metric",
    "test.loss",
    "test.macro_f1",
    "test.micro_f1",
    "test.macro_precision",
    "test.macro_recall",
    "test.micro_precision",
    "test.micro_recall",
    "test.exact_match",
    "test.hamming_loss",
)

MINIMIZE_METRIC_HINTS = ("loss", "hamming", "error", "elapsed", "seconds", "duration", "latency")
MISSING_GRID_VALUE = "(missing)"


@dataclass(frozen=True)
class TrainingPlotPaths:
    loss_plot: Path
    validation_metrics_plot: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "loss_plot": str(self.loss_plot),
            "validation_metrics_plot": str(self.validation_metrics_plot),
        }


@dataclass(frozen=True)
class GridSearchPlotPaths:
    plots_dir: Path
    parameter_metric_plots: dict[str, Path]
    pareto_front_json: Path
    pareto_front_csv: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "plots_dir": str(self.plots_dir),
            "parameter_metric_plots": {
                parameter_name: str(plot_path)
                for parameter_name, plot_path in self.parameter_metric_plots.items()
            },
            "pareto_front_json": str(self.pareto_front_json),
            "pareto_front_csv": str(self.pareto_front_csv),
        }


def plot_static_training_history(
    run_dir: Path | str,
    *,
    validation_metric_names: Sequence[str] | None = None,
    loss_filename: str = "training_validation_loss.png",
    validation_metrics_filename: str = "validation_metrics.png",
) -> TrainingPlotPaths:
    return plot_training_history(
        run_dir,
        title_prefix="Static Pose",
        validation_metric_names=validation_metric_names,
        loss_filename=loss_filename,
        validation_metrics_filename=validation_metrics_filename,
    )


def plot_motion_training_history(
    run_dir: Path | str,
    *,
    validation_metric_names: Sequence[str] | None = None,
    loss_filename: str = "training_validation_loss.png",
    validation_metrics_filename: str = "validation_metrics.png",
) -> TrainingPlotPaths:
    return plot_training_history(
        run_dir,
        title_prefix="Motion Gesture",
        validation_metric_names=validation_metric_names,
        loss_filename=loss_filename,
        validation_metrics_filename=validation_metrics_filename,
    )


def plot_training_history(
    run_dir: Path | str,
    *,
    title_prefix: str,
    validation_metric_names: Sequence[str] | None = None,
    loss_filename: str = "training_validation_loss.png",
    validation_metrics_filename: str = "validation_metrics.png",
) -> TrainingPlotPaths:
    run_path = Path(run_dir)
    metrics_dir = run_path / "metrics"
    train_history = read_metric_history(metrics_dir / "train_metrics.jsonl")
    validation_history = read_metric_history(metrics_dir / "validation_metrics.jsonl")

    loss_plot = metrics_dir / loss_filename
    write_matplotlib_line_plot(
        loss_plot,
        title=f"{title_prefix} Training and Validation Loss",
        x_label="Epoch",
        y_label="Loss",
        series={
            "training loss": metric_series(train_history, "loss"),
            "validation loss": metric_series(validation_history, "loss"),
        },
    )

    metric_names = tuple(validation_metric_names or DEFAULT_VALIDATION_METRICS)
    validation_series = {
        metric_name: metric_series(validation_history, metric_name)
        for metric_name in metric_names
        if metric_series(validation_history, metric_name)
    }
    if not validation_series:
        raise ValueError("no validation metrics were available to plot")

    validation_metrics_plot = metrics_dir / validation_metrics_filename
    write_matplotlib_line_plot(
        validation_metrics_plot,
        title=f"{title_prefix} Validation Metrics by Epoch",
        x_label="Epoch",
        y_label="Metric Value",
        series=validation_series,
        y_min=0.0,
        y_max=1.0,
    )
    return TrainingPlotPaths(loss_plot=loss_plot, validation_metrics_plot=validation_metrics_plot)


def read_metric_history(path: Path | str) -> list[dict[str, object]]:
    history_path = Path(path)
    if not history_path.exists():
        raise FileNotFoundError(f"metric history not found: {history_path}")
    rows: list[dict[str, object]] = []
    with history_path.open("r", encoding="utf-8") as history_file:
        for line_number, line in enumerate(history_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise ValueError(f"metric history row {line_number} is not an object: {history_path}")
            rows.append(dict(payload))
    return sorted(rows, key=lambda row: int(row.get("epoch", 0)))


def metric_series(history: Sequence[Mapping[str, object]], metric_name: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in history:
        metrics = row.get("metrics", {})
        if not isinstance(metrics, Mapping) or metric_name not in metrics:
            continue
        value = metrics[metric_name]
        if not isinstance(value, int | float):
            continue
        points.append((float(row.get("epoch", 0)), float(value)))
    return points


def write_matplotlib_line_plot(
    path: Path | str,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: Mapping[str, Sequence[tuple[float, float]]],
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    non_empty_series = {
        name: sorted(points, key=lambda point: point[0])
        for name, points in series.items()
        if points
    }
    if not non_empty_series:
        raise ValueError("at least one non-empty series is required to write a plot")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(11, 6.5), dpi=120)
    for name, points in non_empty_series.items():
        epochs = [point[0] for point in points]
        values = [point[1] for point in points]
        axis.plot(epochs, values, marker="o", linewidth=2.2, markersize=5, label=name)

    axis.set_title(title, fontsize=16, fontweight="bold", pad=16)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if y_min is not None or y_max is not None:
        axis.set_ylim(bottom=y_min, top=y_max)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, format=plot_format(output_path))
    plt.close(figure)


def plot_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "png"


def plot_static_pose_grid_search_results(
    results_path: Path | str,
    *,
    output_dir: Path | str | None = None,
    metric_names: Sequence[str] | None = None,
    image_format: str = "png",
    dpi: int = 140,
) -> GridSearchPlotPaths:
    results_file = Path(results_path)
    payload = read_grid_search_results(results_file)
    trials = completed_grid_trials(payload)
    if not trials:
        raise ValueError(f"no completed grid-search trials found in {results_file}")

    metrics = tuple(metric_names or collect_grid_metric_names(trials))
    if not metrics:
        raise ValueError(f"no scalar grid-search metrics found in {results_file}")
    parameter_names = varying_loss_parameters(trials)
    if not parameter_names:
        raise ValueError(f"no varying loss_config parameters found in {results_file}")

    metric_directions = grid_metric_directions(trials, metrics)
    pareto_trial_ids = pareto_front_trial_ids(trials, metrics, metric_directions)
    plots_dir = Path(output_dir) if output_dir is not None else results_file.parent / "plots" / "grid_search"
    plots_dir.mkdir(parents=True, exist_ok=True)

    normalized_format = image_format.lower().lstrip(".") or "png"
    parameter_metric_plots: dict[str, Path] = {}
    for parameter_name in parameter_names:
        plot_path = plots_dir / f"grid_search_parameter_{slugify_plot_name(parameter_name)}_metrics.{normalized_format}"
        write_grid_parameter_metric_plot(
            plot_path,
            trials=trials,
            parameter_name=parameter_name,
            metric_names=metrics,
            metric_directions=metric_directions,
            pareto_trial_ids=pareto_trial_ids,
            image_format=normalized_format,
            dpi=dpi,
        )
        parameter_metric_plots[parameter_name] = plot_path

    pareto_front_json = plots_dir / "pareto_front_trials.json"
    pareto_front_csv = plots_dir / "pareto_front_trials.csv"
    write_pareto_front_outputs(
        pareto_front_json,
        pareto_front_csv,
        trials=trials,
        parameter_names=parameter_names,
        metric_names=metrics,
        metric_directions=metric_directions,
        pareto_trial_ids=pareto_trial_ids,
    )
    return GridSearchPlotPaths(
        plots_dir=plots_dir,
        parameter_metric_plots=parameter_metric_plots,
        pareto_front_json=pareto_front_json,
        pareto_front_csv=pareto_front_csv,
    )


def read_grid_search_results(path: Path | str) -> dict[str, Any]:
    results_path = Path(path)
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"grid-search results must be a JSON object: {results_path}")
    return dict(payload)


def completed_grid_trials(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    trials = payload.get("trials")
    if not isinstance(trials, list):
        summary = payload.get("summary", {})
        trials = summary.get("trials") if isinstance(summary, Mapping) else None
    if not isinstance(trials, list):
        raise ValueError("grid-search results must include a trials list")
    return [dict(trial) for trial in trials if isinstance(trial, Mapping) and trial.get("status") == "completed"]


def collect_grid_metric_names(trials: Sequence[Mapping[str, Any]]) -> list[str]:
    available: list[str] = []
    for trial in trials:
        for metric_name in grid_metric_values(trial):
            if metric_name not in available:
                available.append(metric_name)
    ordered = [metric_name for metric_name in DEFAULT_GRID_SEARCH_METRICS if metric_name in available]
    ordered.extend(sorted(metric_name for metric_name in available if metric_name not in ordered))
    return ordered


def grid_metric_values(trial: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in ("score", "best_metric"):
        value = finite_float(trial.get(key))
        if value is not None:
            values[key] = value

    test_metrics = trial.get("test_metrics", {})
    if isinstance(test_metrics, Mapping):
        for key, value in test_metrics.items():
            metric_value = finite_float(value)
            if metric_value is not None:
                values[f"test.{key}"] = metric_value

    summary = trial.get("summary", {})
    if isinstance(summary, Mapping):
        for key, value in summary.items():
            if key in {"best_epoch", "elapsed_seconds"}:
                continue
            metric_value = finite_float(value)
            if metric_value is None:
                continue
            metric_name = key if key not in values else f"summary.{key}"
            values[metric_name] = metric_value
    return values


def finite_float(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def varying_loss_parameters(trials: Sequence[Mapping[str, Any]]) -> list[str]:
    parameter_names = sorted(
        {
            key
            for trial in trials
            if isinstance(trial.get("loss_config"), Mapping)
            for key in trial["loss_config"]
        }
    )
    varying: list[str] = []
    for parameter_name in parameter_names:
        values = {canonical_grid_value(trial_loss_value(trial, parameter_name)) for trial in trials}
        if len(values) > 1:
            varying.append(parameter_name)
    return varying


def trial_loss_value(trial: Mapping[str, Any], parameter_name: str) -> Any:
    loss_config = trial.get("loss_config", {})
    if not isinstance(loss_config, Mapping) or parameter_name not in loss_config:
        return MISSING_GRID_VALUE
    return loss_config[parameter_name]


def canonical_grid_value(value: Any) -> Any:
    if isinstance(value, bool | int | float | str):
        return value
    if value is None:
        return "None"
    return str(value)


def grid_metric_directions(
    trials: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
) -> dict[str, str]:
    directions: dict[str, str] = {}
    for metric_name in metric_names:
        if metric_name in {"score", "best_metric"}:
            mode = first_trial_score_mode(trials)
            directions[metric_name] = mode or infer_grid_metric_direction(metric_name)
            continue
        directions[metric_name] = infer_grid_metric_direction(metric_name)
    return directions


def first_trial_score_mode(trials: Sequence[Mapping[str, Any]]) -> str | None:
    for trial in trials:
        mode = trial.get("score_mode")
        if mode in {"min", "max"}:
            return str(mode)
    return None


def infer_grid_metric_direction(metric_name: str) -> str:
    normalized = metric_name.lower()
    return "min" if any(hint in normalized for hint in MINIMIZE_METRIC_HINTS) else "max"


def pareto_front_trial_ids(
    trials: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
    metric_directions: Mapping[str, str] | None = None,
) -> set[str]:
    directions = dict(metric_directions or grid_metric_directions(trials, metric_names))
    valid_trials = [trial for trial in trials if trial_has_all_metrics(trial, metric_names)]
    front: set[str] = set()
    for candidate in valid_trials:
        dominated = any(
            other is not candidate and trial_dominates(other, candidate, metric_names, directions)
            for other in valid_trials
        )
        if not dominated:
            front.add(str(candidate.get("trial_id", "")))
    return front


def trial_has_all_metrics(trial: Mapping[str, Any], metric_names: Sequence[str]) -> bool:
    values = grid_metric_values(trial)
    return all(metric_name in values for metric_name in metric_names)


def trial_dominates(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    metric_names: Sequence[str],
    metric_directions: Mapping[str, str],
) -> bool:
    left_metrics = grid_metric_values(left)
    right_metrics = grid_metric_values(right)
    strictly_better = False
    for metric_name in metric_names:
        left_value = left_metrics.get(metric_name)
        right_value = right_metrics.get(metric_name)
        if left_value is None or right_value is None:
            return False
        if metric_directions.get(metric_name) == "min":
            if left_value > right_value:
                return False
            strictly_better = strictly_better or left_value < right_value
        else:
            if left_value < right_value:
                return False
            strictly_better = strictly_better or left_value > right_value
    return strictly_better


def write_grid_parameter_metric_plot(
    path: Path | str,
    *,
    trials: Sequence[Mapping[str, Any]],
    parameter_name: str,
    metric_names: Sequence[str],
    metric_directions: Mapping[str, str],
    pareto_trial_ids: set[str],
    image_format: str,
    dpi: int,
) -> None:
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parameter_values = sorted_parameter_values(
        {canonical_grid_value(trial_loss_value(trial, parameter_name)) for trial in trials}
    )
    numeric_axis = all(is_numeric_grid_value(value) for value in parameter_values)
    color_map = plt.get_cmap("tab20" if len(parameter_values) <= 20 else "hsv", max(1, len(parameter_values)))
    value_colors = {value: color_map(index) for index, value in enumerate(parameter_values)}
    x_positions = {value: index for index, value in enumerate(parameter_values)}
    columns = 2 if len(metric_names) <= 8 else 3
    rows = math.ceil(len(metric_names) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 5.8 + 2.0, rows * 3.7), dpi=dpi)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for axis, metric_name in zip(axes_list, metric_names, strict=False):
        all_y_values: list[float] = []
        for parameter_value in parameter_values:
            xs: list[float] = []
            ys: list[float] = []
            for trial_index, trial in enumerate(trials):
                if canonical_grid_value(trial_loss_value(trial, parameter_name)) != parameter_value:
                    continue
                metric_value = grid_metric_values(trial).get(metric_name)
                if metric_value is None:
                    continue
                xs.append(grid_x_value(parameter_value, x_positions, numeric_axis, trial_index, parameter_values))
                ys.append(metric_value)
            if ys:
                axis.scatter(
                    xs,
                    ys,
                    s=44,
                    alpha=0.76,
                    color=value_colors[parameter_value],
                    edgecolors="none",
                )
                all_y_values.extend(ys)

        pareto_points = sorted(
            (
                (
                    grid_x_value(
                        canonical_grid_value(trial_loss_value(trial, parameter_name)),
                        x_positions,
                        numeric_axis,
                        trial_index,
                        parameter_values,
                    ),
                    grid_metric_values(trial).get(metric_name),
                )
                for trial_index, trial in enumerate(trials)
                if str(trial.get("trial_id", "")) in pareto_trial_ids
                and grid_metric_values(trial).get(metric_name) is not None
            ),
            key=lambda point: point[0],
        )
        if pareto_points:
            pareto_xs = [point[0] for point in pareto_points]
            pareto_ys = [float(point[1]) for point in pareto_points]
            axis.scatter(
                pareto_xs,
                pareto_ys,
                s=135,
                marker="*",
                facecolors="white",
                edgecolors="black",
                linewidths=1.2,
                zorder=4,
            )

        axis.set_title(f"{metric_name} ({metric_directions.get(metric_name, 'max')})", fontsize=11, fontweight="bold")
        axis.set_xlabel(parameter_name)
        axis.set_ylabel("Metric value")
        axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.32)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if numeric_axis:
            axis.xaxis.set_major_locator(MaxNLocator(integer=False, nbins=min(8, len(parameter_values))))
        else:
            axis.set_xticks(
                list(x_positions.values()),
                [format_grid_value(value) for value in parameter_values],
                rotation=35,
                ha="right",
            )
        apply_metric_y_limits(axis, all_y_values, metric_name)

    for axis in axes_list[len(metric_names):]:
        axis.set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=value_colors[value],
            markersize=8,
            label=format_grid_value(value),
        )
        for value in parameter_values
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            markerfacecolor="white",
            markersize=12,
            linewidth=0,
            label="Pareto front",
        )
    )
    figure.suptitle(f"Grid Search Metrics by {parameter_name}", fontsize=17, fontweight="bold", y=0.995)
    figure.legend(
        handles=legend_handles,
        title=parameter_name,
        loc="center left",
        bbox_to_anchor=(0.985, 0.5),
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 0.88, 0.965))
    figure.savefig(output_path, format=image_format)
    plt.close(figure)


def sorted_parameter_values(values: set[Any]) -> list[Any]:
    return sorted(values, key=grid_value_sort_key)


def grid_value_sort_key(value: Any) -> tuple[int, float | str]:
    if value == MISSING_GRID_VALUE:
        return (0, "")
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, int | float):
        return (2, float(value))
    return (3, str(value))


def is_numeric_grid_value(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def grid_x_value(
    value: Any,
    x_positions: Mapping[Any, int],
    numeric_axis: bool,
    trial_index: int,
    parameter_values: Sequence[Any],
) -> float:
    base = float(value) if numeric_axis else float(x_positions[value])
    if numeric_axis:
        numeric_values = [float(parameter_value) for parameter_value in parameter_values]
        span = max(numeric_values) - min(numeric_values)
        jitter_scale = span * 0.012 if span > 0 else 0.02
    else:
        jitter_scale = 0.12
    return base + deterministic_jitter(trial_index, jitter_scale)


def deterministic_jitter(index: int, scale: float) -> float:
    return (((index * 17) % 13) - 6) / 6 * scale


def apply_metric_y_limits(axis: Any, values: Sequence[float], metric_name: str) -> None:
    if not values:
        return
    if "loss" not in metric_name.lower() and min(values) >= 0.0 and max(values) <= 1.0:
        axis.set_ylim(-0.03, 1.03)


def write_pareto_front_outputs(
    json_path: Path | str,
    csv_path: Path | str,
    *,
    trials: Sequence[Mapping[str, Any]],
    parameter_names: Sequence[str],
    metric_names: Sequence[str],
    metric_directions: Mapping[str, str],
    pareto_trial_ids: set[str],
) -> None:
    pareto_trials = [trial for trial in trials if str(trial.get("trial_id", "")) in pareto_trial_ids]
    pareto_trials = sorted(pareto_trials, key=lambda trial: str(trial.get("trial_id", "")))
    json_output_path = Path(json_path)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(
        json.dumps(
            {
                "metric_directions": dict(metric_directions),
                "metric_names": list(metric_names),
                "parameter_names": list(parameter_names),
                "trials": [pareto_trial_record(trial, parameter_names, metric_names) for trial in pareto_trials],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    csv_output_path = Path(csv_path)
    columns = ["trial_id", "score", "score_metric", "score_mode"]
    columns.extend(f"loss.{parameter_name}" for parameter_name in parameter_names)
    columns.extend(metric_names)
    columns.append("run_dir")
    with csv_output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for trial in pareto_trials:
            writer.writerow(pareto_trial_record(trial, parameter_names, metric_names))


def pareto_trial_record(
    trial: Mapping[str, Any],
    parameter_names: Sequence[str],
    metric_names: Sequence[str],
) -> dict[str, Any]:
    metrics = grid_metric_values(trial)
    record: dict[str, Any] = {
        "trial_id": trial.get("trial_id"),
        "score": trial.get("score"),
        "score_metric": trial.get("score_metric"),
        "score_mode": trial.get("score_mode"),
        "run_dir": trial.get("run_dir"),
    }
    for parameter_name in parameter_names:
        record[f"loss.{parameter_name}"] = trial_loss_value(trial, parameter_name)
    for metric_name in metric_names:
        record[metric_name] = metrics.get(metric_name)
    return record


def format_grid_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def slugify_plot_name(value: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "parameter"
