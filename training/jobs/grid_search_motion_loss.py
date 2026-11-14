from __future__ import annotations

import argparse
import csv
import itertools
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.common.checkpointing import write_json
from training.common.config import deep_merge, load_yaml_config, write_yaml_config
from training.common.distributed import DistributedContext, distributed_summary, initialize_distributed_context
from training.jobs.train_motion import resolve_project_path, slugify, train_motion


DEFAULT_GRID_SEARCH_CONFIG: dict[str, Any] = {
    "base_config": "training/configs/motion_transformer.yaml",
    "search": {
        "name": "motion_loss_grid",
        "output_root": "runs/motion_loss_grid",
        "trial_output_root": None,
        "selection_metric": None,
        "selection_mode": None,
        "max_trials": None,
        "continue_on_error": False,
        "dry_run": False,
    },
    "trial_overrides": {
        "plots": {"enabled": False},
    },
    "loss_grid": [
        {
            "primary_loss": ["bce"],
            "use_pos_weight": [True, False],
            "phase_weight": [0.25, 0.5, 1.0],
            "direction_weight": [0.0, 0.1, 0.2],
            "displacement_weight": [0.1],
            "velocity_weight": [0.1],
            "duration_weight": [0.05],
            "scale_weight": [0.05],
            "rotation_weight": [0.05],
            "draw_weight": [0.05],
        },
        {
            "primary_loss": ["focal"],
            "use_pos_weight": [True],
            "focal_alpha": [0.25, 0.5],
            "focal_gamma": [1.0, 2.0],
            "phase_weight": [0.25, 0.5],
            "direction_weight": [0.1],
            "displacement_weight": [0.1],
            "velocity_weight": [0.1],
            "duration_weight": [0.05],
            "scale_weight": [0.05],
            "rotation_weight": [0.05],
            "draw_weight": [0.05],
        },
        {
            "primary_loss": ["bce"],
            "use_pos_weight": [True],
            "phase_weight": [0.5],
            "direction_weight": [0.1],
            "displacement_weight": [0.0, 0.1, 0.2],
            "velocity_weight": [0.1],
            "duration_weight": [0.05],
            "scale_weight": [0.05],
            "rotation_weight": [0.05],
            "draw_weight": [0.05],
        },
        {
            "primary_loss": ["bce"],
            "use_pos_weight": [True],
            "phase_weight": [0.5],
            "direction_weight": [0.1],
            "displacement_weight": [0.1],
            "velocity_weight": [0.0, 0.1, 0.2],
            "duration_weight": [0.05],
            "scale_weight": [0.05],
            "rotation_weight": [0.05],
            "draw_weight": [0.05],
        },
        {
            "primary_loss": ["bce"],
            "use_pos_weight": [True],
            "phase_weight": [0.5],
            "direction_weight": [0.1],
            "displacement_weight": [0.1],
            "velocity_weight": [0.1],
            "duration_weight": [0.0, 0.05, 0.1],
            "scale_weight": [0.05],
            "rotation_weight": [0.05],
            "draw_weight": [0.05],
        },
        {
            "primary_loss": ["bce"],
            "use_pos_weight": [True],
            "phase_weight": [0.5],
            "direction_weight": [0.1],
            "displacement_weight": [0.1],
            "velocity_weight": [0.1],
            "duration_weight": [0.05],
            "scale_weight": [0.0, 0.05, 0.1],
            "rotation_weight": [0.05],
            "draw_weight": [0.05],
        },
        {
            "primary_loss": ["bce"],
            "use_pos_weight": [True],
            "phase_weight": [0.5],
            "direction_weight": [0.1],
            "displacement_weight": [0.1],
            "velocity_weight": [0.1],
            "duration_weight": [0.05],
            "scale_weight": [0.05],
            "rotation_weight": [0.0, 0.05, 0.1],
            "draw_weight": [0.05],
        },
        {
            "primary_loss": ["bce"],
            "use_pos_weight": [True],
            "phase_weight": [0.5],
            "direction_weight": [0.1],
            "displacement_weight": [0.1],
            "velocity_weight": [0.1],
            "duration_weight": [0.05],
            "scale_weight": [0.05],
            "rotation_weight": [0.05],
            "draw_weight": [0.0, 0.05, 0.1],
        },
    ],
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a grid search over motion transformer loss hyperparameters.")
    parser.add_argument("--config", type=Path, help="Grid-search YAML config.")
    parser.add_argument("--base-config", type=Path, help="Base motion training YAML config.")
    parser.add_argument("--csv", type=Path, help="Override motion sequence frame CSV path for every trial.")
    parser.add_argument("--output-root", type=Path, help="Root folder for grid-search outputs.")
    parser.add_argument("--trial-output-root", type=Path, help="Root folder for per-trial training run outputs.")
    parser.add_argument("--search-name", help="Human-readable grid-search run name.")
    parser.add_argument("--epochs", type=int, help="Override max epochs for every trial.")
    parser.add_argument("--batch-size", type=int, help="Override batch size for every trial.")
    parser.add_argument("--sequence-length", type=int, help="Override fixed motion window length for every trial.")
    parser.add_argument("--window-stride", type=int, help="Override endpoint stride between generated windows.")
    parser.add_argument("--limit-samples", type=int, help="Use only the first N motion samples for every trial.")
    parser.add_argument("--max-trials", type=int, help="Run only the first N expanded trials.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string.")
    parser.add_argument("--dry-run", action="store_true", help="Run each expanded trial as a trainer dry run.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later trials after a trial fails.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Disable per-trial plot generation.")
    return parser.parse_args(argv)


def run_motion_loss_grid_search(
    config: Mapping[str, Any],
    *,
    device_name: str = "auto",
    dry_run: bool | None = None,
    distributed_context: DistributedContext | None = None,
) -> dict[str, Any]:
    resolved_search_config = deep_merge(DEFAULT_GRID_SEARCH_CONFIG, config)
    search_config = resolved_search_config["search"]
    context = distributed_context or initialize_distributed_context(device_name)
    owns_context = distributed_context is None
    try:
        created_grid_run_dir: Path | None = None
        if context.is_main_process:
            created_grid_run_dir = create_grid_run_dir(search_config)
            write_yaml_config(resolved_search_config, created_grid_run_dir / "grid_search_config.resolved.yaml")
        grid_run_dir = Path(context.broadcast_object(str(created_grid_run_dir)))

        base_config = load_base_training_config(resolved_search_config)
        selection_metric = str(search_config.get("selection_metric") or base_config["training"]["selection_metric"])
        selection_mode = str(search_config.get("selection_mode") or base_config["training"]["selection_mode"])
        trial_dry_run = bool(search_config.get("dry_run", False)) if dry_run is None else dry_run
        score_mode = "min" if trial_dry_run else selection_mode
        continue_on_error = bool(search_config.get("continue_on_error", False))
        max_trials = search_config.get("max_trials")
        trial_output_root = search_config.get("trial_output_root") or grid_run_dir / "trials"

        loss_trials = expand_loss_grid(resolved_search_config["loss_grid"])
        if max_trials is not None:
            loss_trials = loss_trials[: int(max_trials)]
        if not loss_trials:
            raise ValueError("loss grid expanded to zero trials")

        context.print(f"loss grid search run_dir={grid_run_dir} trials={len(loss_trials)}", flush=True)
        start_time = time.time()
        trial_results: list[dict[str, Any]] = []
        for trial_index, loss_config in enumerate(loss_trials, start=1):
            trial_id = f"trial_{trial_index:03d}"
            trial_config = build_trial_config(
                base_config,
                resolved_search_config,
                trial_id,
                loss_config,
                trial_output_root,
            )
            context.print(
                f"trial={trial_index}/{len(loss_trials)} trial_id={trial_id} loss={format_loss_config(loss_config)}",
                flush=True,
            )
            try:
                trial_summary = train_motion(
                    trial_config,
                    device_name=device_name,
                    dry_run=trial_dry_run,
                    distributed_context=context,
                )
                if context.is_main_process:
                    result = build_trial_result(
                        trial_id,
                        loss_config,
                        trial_summary,
                        selection_metric=selection_metric,
                        selection_mode=selection_mode,
                        dry_run=trial_dry_run,
                    )
                    print(
                        f"trial_id={trial_id} status=completed "
                        f"score_metric={result['score_metric']} score={result['score']:.4f} "
                        f"run_dir={result['run_dir']}",
                        flush=True,
                    )
            except Exception as exc:
                if context.is_main_process:
                    result = {
                        "trial_id": trial_id,
                        "status": "failed",
                        "loss_config": dict(loss_config),
                        "score_metric": selection_metric,
                        "score": None,
                        "run_dir": None,
                        "error": str(exc),
                    }
                    print(f"trial_id={trial_id} status=failed error={exc}", flush=True)
                    if not continue_on_error:
                        trial_results.append(result)
                        write_grid_outputs(grid_run_dir, trial_results, None, resolved_search_config)
                if not continue_on_error:
                    raise
            if context.is_main_process:
                trial_results.append(result)

        summary = None
        if context.is_main_process:
            best_trial = select_best_trial(trial_results, score_mode)
            summary = {
                "run_dir": str(grid_run_dir),
                "total_trials": len(trial_results),
                "completed_trials": sum(1 for result in trial_results if result["status"] == "completed"),
                "failed_trials": sum(1 for result in trial_results if result["status"] == "failed"),
                "selection_metric": selection_metric,
                "selection_mode": selection_mode,
                "score_mode": score_mode,
                "elapsed_seconds": time.time() - start_time,
                "best_trial": best_trial,
                "trials": trial_results,
                "distributed": distributed_summary(context),
            }
            write_grid_outputs(grid_run_dir, trial_results, best_trial, summary)
            if best_trial is not None:
                print(
                    f"best trial={best_trial['trial_id']} score_metric={best_trial['score_metric']} "
                    f"score={best_trial['score']:.4f} loss={format_loss_config(best_trial['loss_config'])}",
                    flush=True,
                )
            else:
                print("best trial unavailable: no completed trials", flush=True)
        return context.broadcast_object(summary)
    finally:
        if owns_context:
            context.close()


def load_base_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    base_config_path = resolve_project_path(config["base_config"])
    base_config = load_yaml_config(base_config_path)
    trial_overrides = config.get("trial_overrides", {})
    if not isinstance(trial_overrides, Mapping):
        raise ValueError("trial_overrides must be a mapping")
    return deep_merge(base_config, trial_overrides)


def expand_loss_grid(loss_grid: Any) -> list[dict[str, Any]]:
    if isinstance(loss_grid, Mapping):
        return expand_grid_group(loss_grid)
    if isinstance(loss_grid, Sequence) and not isinstance(loss_grid, str):
        trials: list[dict[str, Any]] = []
        for group in loss_grid:
            if not isinstance(group, Mapping):
                raise ValueError("each loss grid group must be a mapping")
            trials.extend(expand_grid_group(group))
        return trials
    raise ValueError("loss_grid must be a mapping or a list of mappings")


def expand_grid_group(group: Mapping[str, Any]) -> list[dict[str, Any]]:
    keys = list(group.keys())
    value_lists = [normalize_grid_values(key, group[key]) for key in keys]
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*value_lists)]


def normalize_grid_values(key: str, value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        values = list(value)
    else:
        values = [value]
    if not values:
        raise ValueError(f"loss grid key {key!r} has no candidate values")
    return values


def build_trial_config(
    base_config: Mapping[str, Any],
    grid_config: Mapping[str, Any],
    trial_id: str,
    loss_config: Mapping[str, Any],
    trial_output_root: Path | str,
) -> dict[str, Any]:
    base_run_name = str(base_config.get("run", {}).get("name", "motion_transformer"))
    trial_name = f"{base_run_name}_{trial_id}_{loss_slug(loss_config)}"
    return deep_merge(
        base_config,
        {
            "run": {
                "name": trial_name,
                "output_root": str(resolve_project_path(trial_output_root)),
            },
            "loss": dict(loss_config),
            "grid_search": {
                "trial_id": trial_id,
                "search_name": grid_config["search"]["name"],
                "loss_config": dict(loss_config),
            },
        },
    )


def build_trial_result(
    trial_id: str,
    loss_config: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    selection_metric: str,
    selection_mode: str,
    dry_run: bool,
) -> dict[str, Any]:
    score_metric = "dry_run_loss" if dry_run else selection_metric
    if dry_run:
        score = float(summary["loss"])
        effective_mode = "min"
    else:
        score = metric_from_summary(summary, selection_metric)
        effective_mode = selection_mode
    return {
        "trial_id": trial_id,
        "status": "completed",
        "loss_config": dict(loss_config),
        "score_metric": score_metric,
        "score_mode": effective_mode,
        "score": score,
        "run_dir": summary.get("run_dir"),
        "best_epoch": summary.get("best_epoch"),
        "best_metric": summary.get("best_metric"),
        "test_metrics": scalar_metrics(summary.get("test_metrics")),
        "summary": scalar_metrics(summary),
    }


def metric_from_summary(summary: Mapping[str, Any], selection_metric: str) -> float:
    best_metric = summary.get("best_metric")
    if isinstance(best_metric, int | float) and math.isfinite(float(best_metric)):
        return float(best_metric)
    test_metrics = summary.get("test_metrics")
    if isinstance(test_metrics, Mapping):
        value = test_metrics.get(selection_metric)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            return float(value)
    return float("nan")


def scalar_metrics(payload: Any) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    return {key: float(value) for key, value in payload.items() if isinstance(value, int | float)}


def select_best_trial(trial_results: Sequence[Mapping[str, Any]], selection_mode: str) -> dict[str, Any] | None:
    completed = [
        result
        for result in trial_results
        if result["status"] == "completed" and result.get("score") is not None
    ]
    completed = [result for result in completed if math.isfinite(float(result["score"]))]
    if not completed:
        return None
    reverse = selection_mode != "min"
    return dict(sorted(completed, key=lambda result: float(result["score"]), reverse=reverse)[0])


def write_grid_outputs(
    grid_run_dir: Path,
    trial_results: Sequence[Mapping[str, Any]],
    best_trial: Mapping[str, Any] | None,
    summary: Mapping[str, Any],
) -> None:
    write_json(
        grid_run_dir / "grid_search_results.json",
        {
            "summary": dict(summary),
            "best_trial": dict(best_trial) if best_trial is not None else None,
            "trials": [dict(result) for result in trial_results],
        },
    )
    write_grid_results_csv(grid_run_dir / "grid_search_results.csv", trial_results)


def write_grid_results_csv(path: Path | str, trial_results: Sequence[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [flatten_trial_result(result) for result in trial_results]
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def flatten_trial_result(result: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "trial_id": result.get("trial_id"),
        "status": result.get("status"),
        "score_metric": result.get("score_metric"),
        "score_mode": result.get("score_mode"),
        "score": result.get("score"),
        "best_epoch": result.get("best_epoch"),
        "best_metric": result.get("best_metric"),
        "run_dir": result.get("run_dir"),
        "error": result.get("error"),
    }
    for key, value in dict(result.get("loss_config") or {}).items():
        row[f"loss.{key}"] = value
    for key, value in dict(result.get("test_metrics") or {}).items():
        row[f"test.{key}"] = value
    return row


def create_grid_run_dir(search_config: Mapping[str, Any]) -> Path:
    output_root = resolve_project_path(search_config["output_root"])
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_name = str(search_config.get("name") or "motion_loss_grid")
    run_dir = output_root / f"{timestamp}_{slugify(run_name)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def loss_slug(loss_config: Mapping[str, Any]) -> str:
    parts = [f"{key}_{loss_config[key]}" for key in sorted(loss_config)]
    return slugify("_".join(parts))[:96]


def format_loss_config(loss_config: Mapping[str, Any]) -> str:
    return ",".join(f"{key}={value}" for key, value in sorted(loss_config.items()))


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml_config(args.config)
    overrides: dict[str, Any] = {}
    if args.base_config is not None:
        overrides["base_config"] = str(args.base_config)
    if args.output_root is not None:
        overrides.setdefault("search", {})["output_root"] = str(args.output_root)
    if args.trial_output_root is not None:
        overrides.setdefault("search", {})["trial_output_root"] = str(args.trial_output_root)
    if args.search_name is not None:
        overrides.setdefault("search", {})["name"] = args.search_name
    if args.max_trials is not None:
        overrides.setdefault("search", {})["max_trials"] = args.max_trials
    if args.dry_run:
        overrides.setdefault("search", {})["dry_run"] = True
    if args.continue_on_error:
        overrides.setdefault("search", {})["continue_on_error"] = True
    if args.csv is not None:
        overrides.setdefault("trial_overrides", {}).setdefault("data", {})["csv_path"] = str(args.csv)
    if args.epochs is not None:
        overrides.setdefault("trial_overrides", {}).setdefault("training", {})["max_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides.setdefault("trial_overrides", {}).setdefault("data", {})["batch_size"] = args.batch_size
    if args.sequence_length is not None:
        overrides.setdefault("trial_overrides", {}).setdefault("data", {})["sequence_length"] = args.sequence_length
    if args.window_stride is not None:
        overrides.setdefault("trial_overrides", {}).setdefault("data", {})["window_stride"] = args.window_stride
    if args.limit_samples is not None:
        overrides.setdefault("trial_overrides", {}).setdefault("data", {})["limit_samples"] = args.limit_samples
    if args.no_plots:
        overrides.setdefault("trial_overrides", {}).setdefault("plots", {})["enabled"] = False
    return deep_merge(config, overrides)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    summary = run_motion_loss_grid_search(config, device_name=args.device, dry_run=args.dry_run or None)
    print(f"grid_run_dir={summary['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
