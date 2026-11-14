from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from training.common.plotting import plot_static_pose_grid_search_results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot static-pose grid-search metrics by varied loss parameter."
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to grid_search_results.json from training.jobs.grid_search_static_pose_loss.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for generated charts and Pareto-front exports.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Optional metric names to plot. Defaults to all scalar score/test metrics found in the results.",
    )
    parser.add_argument(
        "--format",
        default="png",
        help="Matplotlib image format, such as png, svg, or pdf.",
    )
    parser.add_argument(
        "--dpi", type=int, default=140, help="Output DPI for raster formats."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    argv = [
        "--results",
        "runs/static_pose_loss_grid/20260721-204029_static_pose_loss_grid/grid_search_results.json",
    ]

    args = parse_args(argv)
    plot_paths = plot_static_pose_grid_search_results(
        args.results,
        output_dir=args.output_dir,
        metric_names=args.metrics,
        image_format=args.format,
        dpi=args.dpi,
    )
    print(f"plots_dir={plot_paths.plots_dir}")
    print(f"pareto_front_json={plot_paths.pareto_front_json}")
    print(f"pareto_front_csv={plot_paths.pareto_front_csv}")
    for parameter_name, plot_path in plot_paths.parameter_metric_plots.items():
        print(f"parameter_plot.{parameter_name}={plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
