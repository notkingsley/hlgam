"""Command line entrypoint for Shadeform remote training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from shadeform.config import (
    DEFAULT_ENV_FILE,
    load_env_values,
    load_settings,
    parse_env_assignments,
)
from shadeform.runner import (
    DEFAULT_REMOTE_WORKDIR,
    PROJECT_ROOT,
    build_remote_training_plan,
    run_remote_training,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HLGAM training jobs on Shadeform GPUs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser(
        "train", help="Provision a GPU and run a static-pose or motion training job."
    )
    train.add_argument(
        "--model-family", choices=("static_pose", "motion"), required=True
    )
    train.add_argument(
        "--job",
        choices=("train", "loss_grid"),
        default="train",
        help="Training job to run. Use loss_grid for the static or motion loss hyperparameter grid search.",
    )
    train.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    train.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    train.add_argument(
        "--plan",
        action="store_true",
        help="Print the remote execution plan without calling Shadeform.",
    )
    train.add_argument(
        "--gpu-type", help="Shadeform GPU type filter, e.g. A100, L40S, A10G."
    )
    train.add_argument(
        "--num-gpus",
        type=int,
        help="Number of GPUs to request. Values greater than one launch single-node DDP training.",
    )
    train.add_argument(
        "--cloud", help="Optional Shadeform cloud filter or direct provider value."
    )
    train.add_argument(
        "--region", help="Optional Shadeform region filter or direct provider value."
    )
    train.add_argument(
        "--shade-instance-type", help="Optional direct Shadeform instance type."
    )
    train.add_argument(
        "--yes",
        action="store_true",
        help="Select the cheapest matching provider without prompting.",
    )
    train.add_argument(
        "--wait-unavailable",
        action="store_true",
        help="Poll the cheapest unavailable provider if none are available.",
    )
    train.add_argument(
        "--keep-instance",
        action="store_true",
        help="Do not delete the instance after the run.",
    )
    train.add_argument(
        "--keep-instance-on-failure",
        action="store_true",
        help="Keep failed instances for SSH debugging.",
    )
    train.add_argument("--remote-workdir", default=DEFAULT_REMOTE_WORKDIR)
    train.add_argument(
        "--config", type=Path, help="Training config to upload and pass to the trainer."
    )
    train.add_argument(
        "--csv",
        type=Path,
        help="Project-relative path where the Drive-downloaded landmark CSV is placed remotely.",
    )
    train.add_argument(
        "--gdrive-csv-file-id",
        help="Required Google Drive file ID for the input CSV. Falls back to SHADEFORM_GDRIVE_*_CSV_FILE_ID.",
    )
    train.add_argument(
        "--gdrive-results-folder-id",
        help="Required Google Drive folder ID where completed run folders are uploaded after training.",
    )
    train.add_argument(
        "--gdrive-service-account-key",
        type=Path,
        help="Local Google service-account JSON key used for Drive transfers.",
    )
    train.add_argument(
        "--output-root",
        help="Remote project-relative output root. Defaults to the model family run folder.",
    )
    train.add_argument(
        "--run-name", help="Run name for trainers, or search name for loss_grid jobs."
    )
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument(
        "--limit-rows", type=int, help="Static-pose smoke-test row limit."
    )
    train.add_argument(
        "--limit-samples", type=int, help="Motion smoke-test sample limit."
    )
    train.add_argument(
        "--sequence-length", type=int, help="Motion sequence length override."
    )
    train.add_argument(
        "--window-stride", type=int, help="Motion window stride override."
    )
    train.add_argument("--max-trials", type=int, help="Loss-grid trial cap.")
    train.add_argument(
        "--trial-output-root",
        help="Remote project-relative root for per-trial grid runs.",
    )
    train.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue later loss-grid trials after a trial failure.",
    )
    train.add_argument("--device", default="auto")
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the trainer's dry-run mode remotely.",
    )
    train.add_argument("--no-plots", action="store_true")
    train.add_argument(
        "--dep", dest="extra_upload_paths", type=Path, action="append", default=[]
    )
    train.add_argument("--return", dest="return_paths", action="append", default=[])
    train.add_argument(
        "--env",
        dest="remote_env",
        action="append",
        default=[],
        help="Remote env var assignment KEY=VALUE.",
    )
    train.add_argument(
        "--setup-command",
        action="append",
        default=[],
        help="Extra remote setup command after dependency install.",
    )
    train.add_argument(
        "training_args",
        nargs=argparse.REMAINDER,
        help="Additional trainer args after '--'.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        try:
            return run_train_command(args)
        except ValueError as exc:
            parser.error(str(exc))
    parser.error(f"unsupported command: {args.command}")
    return 2


def run_train_command(args: argparse.Namespace) -> int:
    remote_env = parse_env_assignments(args.remote_env)
    extra_training_args = tuple(
        argument for argument in args.training_args if argument != "--"
    )
    plan_kwargs = {
        "training_job": args.job,
        "config_path": args.config,
        "csv_path": args.csv,
        "output_root": args.output_root,
        "run_name": args.run_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "limit_rows": args.limit_rows,
        "limit_samples": args.limit_samples,
        "sequence_length": args.sequence_length,
        "window_stride": args.window_stride,
        "max_trials": args.max_trials,
        "trial_output_root": args.trial_output_root,
        "continue_on_error": args.continue_on_error,
        "no_plots": args.no_plots,
        "dry_run": args.dry_run,
        "device": args.device,
        "remote_workdir": args.remote_workdir,
        "extra_upload_paths": tuple(args.extra_upload_paths),
        "return_paths": tuple(args.return_paths),
        "remote_env": remote_env,
        "extra_setup_commands": tuple(args.setup_command),
        "extra_training_args": extra_training_args,
        "gdrive_csv_file_id": args.gdrive_csv_file_id,
        "gdrive_results_folder_id": args.gdrive_results_folder_id,
    }
    apply_gdrive_defaults(args, plan_kwargs)
    if args.plan:
        plan = build_remote_training_plan(
            model_family=args.model_family,
            project_root=args.project_root,
            num_gpus=resolve_plan_num_gpus(args),
            **plan_kwargs,
        )
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return 0

    settings = load_settings(args.env_file)
    summary = run_remote_training(
        settings=settings,
        model_family=args.model_family,
        project_root=args.project_root,
        gpu_type=args.gpu_type,
        num_gpus=args.num_gpus,
        cloud=args.cloud,
        region=args.region,
        shade_instance_type=args.shade_instance_type,
        auto_select=args.yes,
        wait_for_unavailable=args.wait_unavailable,
        keep_instance=args.keep_instance,
        keep_instance_on_failure=args.keep_instance_on_failure,
        gdrive_service_account_key_path=args.gdrive_service_account_key,
        **plan_kwargs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def apply_gdrive_defaults(
    args: argparse.Namespace, plan_kwargs: dict[str, object]
) -> None:
    values = load_env_values(args.env_file)
    if not plan_kwargs.get("gdrive_csv_file_id"):
        plan_kwargs["gdrive_csv_file_id"] = env_gdrive_csv_file_id(
            values, args.model_family
        )
    if not plan_kwargs.get("gdrive_results_folder_id"):
        plan_kwargs["gdrive_results_folder_id"] = first_env_value(
            values,
            "SHADEFORM_GDRIVE_RESULTS_FOLDER_ID",
            "GDRIVE_RESULTS_FOLDER_ID",
        )


def resolve_plan_num_gpus(args: argparse.Namespace) -> int | None:
    if args.num_gpus is not None:
        return args.num_gpus
    value = first_env_value(load_env_values(args.env_file), "SHADEFORM_NUM_GPUS")
    return int(value) if value is not None else None


def env_gdrive_csv_file_id(values: dict[str, str], model_family: str) -> str | None:
    if model_family == "static_pose":
        value = first_env_value(values, "SHADEFORM_GDRIVE_STATIC_POSE_CSV_FILE_ID")
        if value:
            return value
    if model_family == "motion":
        value = first_env_value(values, "SHADEFORM_GDRIVE_MOTION_CSV_FILE_ID")
        if value:
            return value
    return first_env_value(values, "SHADEFORM_GDRIVE_CSV_FILE_ID")


def first_env_value(values: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = values.get(key)
        if value is not None and value.strip():
            return value.strip()
    return None


if __name__ == "__main__":
    argv = [
        "train",
        "--num-gpus",
        "1",
        "--model-family",
        "motion",
        "--job",
        "loss_grid",
        # "--plan",
    ]
    raise SystemExit(main(argv))
