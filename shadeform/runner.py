"""Remote Shadeform orchestration for HLGAM training jobs."""

from __future__ import annotations

import hashlib
import json
import secrets
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from shadeform.api import AvailabilityOption, ShadeformAPI
from shadeform.config import ShadeformSettings
from shadeform.ssh import run_remote, upload, wait_for_ssh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE_WORKDIR = "/home/shadeform/gesture-app"
REMOTE_PYTHON_VERSION = "3.12"
UV_ENV = "source $HOME/.local/bin/env"
GPU_TORCH_PACKAGES = ("matplotlib>=3.8.0", "torch>=2.3.0")
GDRIVE_HELPER_PATH = Path("shadeform/gdrive.py")
REMOTE_GDRIVE_SERVICE_ACCOUNT_KEY_PATH = "/tmp/shadeform_gdrive_service_account.json"


@dataclass(frozen=True)
class TrainingTarget:
    model_family: str
    training_job: str
    entrypoint: str
    default_config_path: Path
    default_csv_path: Path
    default_output_root: str
    limit_argument: str
    run_name_argument: str
    supports_sequence_args: bool
    supports_grid_args: bool


TRAINING_TARGETS = {
    ("static_pose", "train"): TrainingTarget(
        model_family="static_pose",
        training_job="train",
        entrypoint="training.jobs.train_static_pose",
        default_config_path=Path("training/configs/static_pose_transformer.yaml"),
        default_csv_path=Path("data/static_poses/static_pose_frames.csv"),
        default_output_root="runs/static_pose",
        limit_argument="--limit-rows",
        run_name_argument="--run-name",
        supports_sequence_args=False,
        supports_grid_args=False,
    ),
    ("static_pose", "loss_grid"): TrainingTarget(
        model_family="static_pose",
        training_job="loss_grid",
        entrypoint="training.jobs.grid_search_static_pose_loss",
        default_config_path=Path("training/configs/static_pose_loss_grid.yaml"),
        default_csv_path=Path("data/static_poses/static_pose_frames.csv"),
        default_output_root="runs/static_pose_loss_grid",
        limit_argument="--limit-rows",
        run_name_argument="--search-name",
        supports_sequence_args=False,
        supports_grid_args=True,
    ),
    ("motion", "train"): TrainingTarget(
        model_family="motion",
        training_job="train",
        entrypoint="training.jobs.train_motion",
        default_config_path=Path("training/configs/motion_transformer.yaml"),
        default_csv_path=Path("data/motion_sequences/motion_sequence_frames.csv"),
        default_output_root="runs/motion",
        limit_argument="--limit-samples",
        run_name_argument="--run-name",
        supports_sequence_args=True,
        supports_grid_args=False,
    ),
    ("motion", "loss_grid"): TrainingTarget(
        model_family="motion",
        training_job="loss_grid",
        entrypoint="training.jobs.grid_search_motion_loss",
        default_config_path=Path("training/configs/motion_loss_grid.yaml"),
        default_csv_path=Path("data/motion_sequences/motion_sequence_frames.csv"),
        default_output_root="runs/motion_loss_grid",
        limit_argument="--limit-samples",
        run_name_argument="--search-name",
        supports_sequence_args=True,
        supports_grid_args=True,
    ),
}


@dataclass(frozen=True)
class UploadSpec:
    local_path: Path
    remote_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "local_path": str(self.local_path),
            "remote_path": self.remote_path,
        }


@dataclass(frozen=True)
class RemoteTrainingPlan:
    model_family: str
    training_job: str
    num_gpus: int
    entrypoint: str
    instance_name: str
    remote_workdir: str
    upload_paths: tuple[UploadSpec, ...]
    return_paths: tuple[str, ...]
    setup_commands: tuple[str, ...]
    training_command: str
    gdrive: GoogleDrivePlan
    pre_training_commands: tuple[str, ...] = ()
    post_training_commands: tuple[str, ...] = ()
    remote_env: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "training_job": self.training_job,
            "num_gpus": self.num_gpus,
            "entrypoint": self.entrypoint,
            "instance_name": self.instance_name,
            "remote_workdir": self.remote_workdir,
            "upload_paths": [upload_path.to_dict() for upload_path in self.upload_paths],
            "return_paths": list(self.return_paths),
            "setup_commands": list(self.setup_commands),
            "pre_training_commands": list(self.pre_training_commands),
            "training_command": self.training_command,
            "post_training_commands": list(self.post_training_commands),
            "remote_env": dict(self.remote_env),
            "gdrive": self.gdrive.to_dict(),
        }

    @property
    def needs_gdrive(self) -> bool:
        return True


@dataclass(frozen=True)
class GoogleDrivePlan:
    service_account_remote_path: str
    input_csv_file_id: str
    input_csv_remote_path: str
    results_folder_id: str
    results_subfolder: str

    def to_dict(self) -> dict[str, str]:
        return {
            "service_account_remote_path": self.service_account_remote_path,
            "input_csv_file_id": self.input_csv_file_id,
            "input_csv_remote_path": self.input_csv_remote_path,
            "results_folder_id": self.results_folder_id,
            "results_subfolder": self.results_subfolder,
        }


def build_remote_training_plan(
    *,
    model_family: str,
    training_job: str = "train",
    project_root: Path | str = PROJECT_ROOT,
    config_path: Path | str | None = None,
    csv_path: Path | str | None = None,
    output_root: Path | str | None = None,
    run_name: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    limit_rows: int | None = None,
    limit_samples: int | None = None,
    sequence_length: int | None = None,
    window_stride: int | None = None,
    max_trials: int | None = None,
    trial_output_root: Path | str | None = None,
    continue_on_error: bool = False,
    no_plots: bool = False,
    dry_run: bool = False,
    device: str = "auto",
    num_gpus: int | None = None,
    remote_workdir: str = DEFAULT_REMOTE_WORKDIR,
    extra_upload_paths: Sequence[Path | str] = (),
    return_paths: Sequence[str] = (),
    remote_env: Mapping[str, str] | None = None,
    extra_setup_commands: Sequence[str] = (),
    extra_training_args: Sequence[str] = (),
    gdrive_csv_file_id: str | None = None,
    gdrive_results_folder_id: str | None = None,
    gdrive_service_account_remote_path: str = REMOTE_GDRIVE_SERVICE_ACCOUNT_KEY_PATH,
    instance_name: str | None = None,
    validate_paths: bool = True,
) -> RemoteTrainingPlan:
    target = target_for_model_family(model_family, training_job)
    root = Path(project_root).resolve()
    resolved_num_gpus = normalize_num_gpus(num_gpus)
    resolved_gdrive_csv_file_id = require_nonempty_text(
        gdrive_csv_file_id,
        "gdrive_csv_file_id",
        "Shadeform training always downloads the input CSV from Google Drive. "
        "Pass --gdrive-csv-file-id or set SHADEFORM_GDRIVE_CSV_FILE_ID, "
        "SHADEFORM_GDRIVE_STATIC_POSE_CSV_FILE_ID, or SHADEFORM_GDRIVE_MOTION_CSV_FILE_ID.",
    )
    resolved_gdrive_results_folder_id = require_nonempty_text(
        gdrive_results_folder_id,
        "gdrive_results_folder_id",
        "Shadeform training always uploads run results to Google Drive. "
        "Pass --gdrive-results-folder-id or set SHADEFORM_GDRIVE_RESULTS_FOLDER_ID.",
    )
    resolved_config_path = resolve_local_path(config_path or target.default_config_path, root)
    resolved_csv_path = resolve_local_path(csv_path or target.default_csv_path, root)
    resolved_output_root = normalize_remote_relative_path(output_root or target.default_output_root)
    job_slug = "training" if target.training_job == "train" else target.training_job.replace("_", "-")
    resolved_instance_name = instance_name or f"hlgam-{target.model_family.replace('_', '-')}-{job_slug}"
    resolved_instance_name += f"-{secrets.token_hex(4)}"

    upload_specs = build_upload_specs(
        root,
        remote_workdir=remote_workdir,
        config_path=resolved_config_path,
        csv_path=resolved_csv_path,
        extra_upload_paths=extra_upload_paths,
        include_csv_upload=False,
        include_gdrive_helper=True,
        validate_paths=validate_paths,
    )
    config_argument = remote_argument_for_local_path(resolved_config_path, root, remote_workdir)
    csv_argument = remote_argument_for_local_path(resolved_csv_path, root, remote_workdir)
    training_command = build_training_command(
        target=target,
        config_argument=config_argument,
        csv_argument=csv_argument,
        output_root=resolved_output_root,
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        limit_rows=limit_rows,
        limit_samples=limit_samples,
        sequence_length=sequence_length,
        window_stride=window_stride,
        max_trials=max_trials,
        trial_output_root=trial_output_root,
        continue_on_error=continue_on_error,
        no_plots=no_plots,
        dry_run=dry_run,
        device=device,
        num_gpus=resolved_num_gpus,
        remote_workdir=remote_workdir,
        remote_env=remote_env or {},
        extra_training_args=extra_training_args,
    )
    setup_commands = (
        "command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh",
        f"{UV_ENV} && uv python install {REMOTE_PYTHON_VERSION}",
        f"{UV_ENV} && cd {shlex.quote(remote_workdir)} && uv sync --python {REMOTE_PYTHON_VERSION} --no-dev",
        f"{UV_ENV} && cd {shlex.quote(remote_workdir)} && "
        f"uv pip install {' '.join(shlex.quote(package) for package in GPU_TORCH_PACKAGES)} --torch-backend=auto",
        *tuple(extra_setup_commands),
    )
    resolved_return_paths = unique_paths((resolved_output_root, *return_paths))
    gdrive_plan = build_gdrive_plan(
        input_csv_file_id=resolved_gdrive_csv_file_id,
        input_csv_remote_path=remote_join(remote_workdir, csv_argument),
        results_folder_id=resolved_gdrive_results_folder_id,
        results_subfolder=resolved_instance_name,
        service_account_remote_path=gdrive_service_account_remote_path,
    )
    pre_training_commands = build_pre_training_commands(gdrive_plan, remote_workdir)
    post_training_commands = build_post_training_commands(
        gdrive_plan,
        remote_workdir,
        return_paths=resolved_return_paths,
    )
    return RemoteTrainingPlan(
        model_family=target.model_family,
        training_job=target.training_job,
        num_gpus=resolved_num_gpus,
        entrypoint=target.entrypoint,
        instance_name=resolved_instance_name,
        remote_workdir=remote_workdir,
        upload_paths=tuple(upload_specs),
        return_paths=resolved_return_paths,
        setup_commands=setup_commands,
        training_command=training_command,
        gdrive=gdrive_plan,
        pre_training_commands=pre_training_commands,
        post_training_commands=post_training_commands,
        remote_env=dict(remote_env or {}),
    )


def run_remote_training(
    *,
    settings: ShadeformSettings,
    model_family: str,
    project_root: Path | str = PROJECT_ROOT,
    gpu_type: str | None = None,
    num_gpus: int | None = None,
    cloud: str | None = None,
    region: str | None = None,
    shade_instance_type: str | None = None,
    auto_select: bool = False,
    wait_for_unavailable: bool = False,
    keep_instance: bool = False,
    keep_instance_on_failure: bool = False,
    **plan_kwargs: Any,
) -> dict[str, Any]:
    settings.validate()
    gdrive_csv_file_id = plan_kwargs.pop("gdrive_csv_file_id", None)
    gdrive_results_folder_id = plan_kwargs.pop("gdrive_results_folder_id", None)
    gdrive_service_account_key_path = plan_kwargs.pop("gdrive_service_account_key_path", None)
    gdrive_csv_file_id = resolve_required_gdrive_csv_file_id(settings, model_family, gdrive_csv_file_id)
    gdrive_results_folder_id = resolve_required_gdrive_results_folder_id(settings, gdrive_results_folder_id)
    requested_num_gpus = normalize_num_gpus(num_gpus if num_gpus is not None else settings.default_num_gpus)
    plan = build_remote_training_plan(
        model_family=model_family,
        project_root=project_root,
        num_gpus=requested_num_gpus,
        gdrive_csv_file_id=gdrive_csv_file_id,
        gdrive_results_folder_id=gdrive_results_folder_id,
        **plan_kwargs,
    )
    service_account_key_path = resolve_gdrive_service_account_key_path(
        settings,
        gdrive_service_account_key_path,
    )
    api = ShadeformAPI(settings.api_key)
    provider = resolve_provider(
        api,
        gpu_type=gpu_type or settings.default_gpu_type,
        num_gpus=requested_num_gpus,
        cloud=cloud or settings.default_cloud,
        region=region or settings.default_region,
        shade_instance_type=shade_instance_type or settings.default_shade_instance_type,
        auto_select=auto_select,
        wait_for_unavailable=wait_for_unavailable,
    )

    instance_id: str | None = None
    instance_info: dict[str, Any] | None = None
    succeeded = False
    downloaded_paths: list[str] = []
    uploaded_drive_paths: list[str] = []
    deleted_instance = False
    ssh_user: str | None = None
    ip: str | None = None
    port: int | None = None
    try:
        print("Creating Shadeform instance ...", file=sys.stderr)
        instance_id = api.create_instance(
            cloud=provider.cloud,
            region=provider.region,
            shade_instance_type=provider.shade_instance_type,
            name=plan.instance_name,
        )
        info = api.wait_until_active(instance_id)
        instance_info = {
            "id": info.id,
            "ip": info.ip,
            "ssh_user": info.ssh_user,
            "ssh_port": info.ssh_port,
            "cloud": info.cloud,
            "region": info.region,
            "shade_instance_type": info.shade_instance_type,
        }
        assert info.ip is not None
        ip = info.ip
        ssh_user = info.ssh_user or settings.ssh_user
        port = info.ssh_port
        wait_for_ssh(key_path=settings.ssh_key_path, user=ssh_user, ip=ip, port=port)
        run_remote(
            key_path=settings.ssh_key_path,
            user=ssh_user,
            ip=ip,
            port=port,
            command=f"mkdir -p {shlex.quote(plan.remote_workdir)}",
        )
        upload_training_payload(settings, plan, ssh_user=ssh_user, ip=ip, port=port)
        upload_gdrive_service_account(
            settings,
            plan,
            service_account_key_path=service_account_key_path,
            ssh_user=ssh_user,
            ip=ip,
            port=port,
        )
        for command in plan.setup_commands:
            run_remote(key_path=settings.ssh_key_path, user=ssh_user, ip=ip, port=port, command=command)
        for command in plan.pre_training_commands:
            run_remote(key_path=settings.ssh_key_path, user=ssh_user, ip=ip, port=port, command=command)

        try:
            run_remote(
                key_path=settings.ssh_key_path,
                user=ssh_user,
                ip=ip,
                port=port,
                command=plan.training_command,
            )
        finally:
            for command in plan.post_training_commands:
                run_remote(key_path=settings.ssh_key_path, user=ssh_user, ip=ip, port=port, command=command)
            uploaded_drive_paths = planned_drive_upload_paths(plan)
        succeeded = True
    finally:
        downloaded_paths = download_return_paths_from_gdrive(
            settings,
            plan,
            service_account_key_path=service_account_key_path,
        )
        if ssh_user and ip and port:
            cleanup_gdrive_service_account(settings, plan, ssh_user=ssh_user, ip=ip, port=port)
        should_delete = instance_id is not None and not keep_instance and (succeeded or not keep_instance_on_failure)
        if should_delete:
            print(f"Deleting Shadeform instance {instance_id} ...", file=sys.stderr)
            api.delete_instance(instance_id)
            deleted_instance = True
    return {
        "succeeded": succeeded,
        "instance_id": instance_id,
        "instance": instance_info,
        "provider": provider_to_dict(provider),
        "plan": plan.to_dict(),
        "downloaded_paths": downloaded_paths,
        "uploaded_drive_paths": uploaded_drive_paths,
        "deleted_instance": deleted_instance,
    }


def upload_training_payload(
    settings: ShadeformSettings,
    plan: RemoteTrainingPlan,
    *,
    ssh_user: str,
    ip: str,
    port: int,
) -> None:
    for upload_spec in plan.upload_paths:
        print(f"Uploading {upload_spec.local_path} -> {upload_spec.remote_path}", file=sys.stderr)
        upload(
            key_path=settings.ssh_key_path,
            user=ssh_user,
            ip=ip,
            port=port,
            local_path=upload_spec.local_path,
            remote_path=upload_spec.remote_path,
        )


def download_return_paths_from_gdrive(
    settings: ShadeformSettings,
    plan: RemoteTrainingPlan,
    *,
    service_account_key_path: Path,
) -> list[str]:
    destination_root = settings.results_dir / plan.instance_name
    command = build_gdrive_download_command(
        service_account_key_path=service_account_key_path,
        parent_folder_id=plan.gdrive.results_folder_id,
        subfolder_name=plan.gdrive.results_subfolder,
        destination_root=destination_root,
        return_paths=plan.return_paths,
    )
    print(f"Downloading Shadeform returns from Google Drive -> {destination_root}", file=sys.stderr)
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, check=False)
    if result.returncode != 0:
        print(f"  Google Drive download exited with code {result.returncode}", file=sys.stderr)
        return []
    return [str(destination_root / return_path) for return_path in plan.return_paths]


def resolve_gdrive_service_account_key_path(
    settings: ShadeformSettings,
    override: Path | str | None,
) -> Path:
    if override is None:
        return settings.require_gdrive_service_account_key_path()
    resolved = Path(override).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Google Drive service-account key does not exist: {resolved}")
    return resolved


def resolve_required_gdrive_csv_file_id(
    settings: ShadeformSettings,
    model_family: str,
    override: str | None,
) -> str:
    value = override or settings.gdrive_input_csv_file_id_for(model_family)
    return require_nonempty_text(
        value,
        "gdrive_csv_file_id",
        "Set --gdrive-csv-file-id or configure a model-family-specific "
        "SHADEFORM_GDRIVE_*_CSV_FILE_ID value in shadeform.env.",
    )


def resolve_required_gdrive_results_folder_id(
    settings: ShadeformSettings,
    override: str | None,
) -> str:
    return require_nonempty_text(
        override or settings.gdrive_results_folder_id,
        "gdrive_results_folder_id",
        "Set --gdrive-results-folder-id or SHADEFORM_GDRIVE_RESULTS_FOLDER_ID in shadeform.env.",
    )


def upload_gdrive_service_account(
    settings: ShadeformSettings,
    plan: RemoteTrainingPlan,
    *,
    service_account_key_path: Path,
    ssh_user: str,
    ip: str,
    port: int,
) -> None:
    assert plan.gdrive is not None
    upload(
        key_path=settings.ssh_key_path,
        user=ssh_user,
        ip=ip,
        port=port,
        local_path=service_account_key_path,
        remote_path=plan.gdrive.service_account_remote_path,
    )
    run_remote(
        key_path=settings.ssh_key_path,
        user=ssh_user,
        ip=ip,
        port=port,
        command=f"chmod 600 {shlex.quote(plan.gdrive.service_account_remote_path)}",
    )


def cleanup_gdrive_service_account(
    settings: ShadeformSettings,
    plan: RemoteTrainingPlan,
    *,
    ssh_user: str,
    ip: str,
    port: int,
) -> None:
    if plan.gdrive is None:
        return
    run_remote(
        key_path=settings.ssh_key_path,
        user=ssh_user,
        ip=ip,
        port=port,
        command=f"rm -f {shlex.quote(plan.gdrive.service_account_remote_path)}",
        check=False,
    )


def resolve_provider(
    api: ShadeformAPI,
    *,
    gpu_type: str,
    num_gpus: int,
    cloud: str | None,
    region: str | None,
    shade_instance_type: str | None,
    auto_select: bool,
    wait_for_unavailable: bool,
) -> AvailabilityOption:
    if cloud and region and shade_instance_type:
        return direct_provider_option(
            cloud=cloud,
            region=region,
            shade_instance_type=shade_instance_type,
            gpu_type=gpu_type,
            num_gpus=num_gpus,
        )

    options = filter_options(
        api.list_available(gpu_type, num_gpus),
        cloud=cloud,
        region=region,
        shade_instance_type=shade_instance_type,
    )
    if options:
        return options[0] if auto_select or len(options) == 1 else prompt_provider_choice(options)
    if not wait_for_unavailable:
        raise RuntimeError("No available Shadeform providers matched the requested GPU constraints")

    unavailable = filter_options(
        api.list_unavailable(gpu_type, num_gpus),
        cloud=cloud,
        region=region,
        shade_instance_type=shade_instance_type,
    )
    if not unavailable:
        raise RuntimeError("No unavailable Shadeform providers matched the requested GPU constraints")
    selected = unavailable[0] if auto_select or len(unavailable) == 1 else prompt_provider_choice(unavailable)
    return api.wait_for_availability(selected)


def prompt_provider_choice(options: Sequence[AvailabilityOption]) -> AvailabilityOption:
    print_options_table(options)
    while True:
        try:
            choice = int(input(f"Select provider [1-{len(options)}]: "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
        except (EOFError, ValueError):
            pass
        print("Invalid provider choice", file=sys.stderr)


def print_options_table(options: Sequence[AvailabilityOption]) -> None:
    print("\nAvailable Shadeform providers:\n", file=sys.stderr)
    header = f"  {'#':>3}  {'Cloud':<16} {'Region':<24} {'Type':<16} {'GPU':<12} {'$/hr':>7}"
    print(header, file=sys.stderr)
    print("  " + "-" * (len(header) - 2), file=sys.stderr)
    for index, option in enumerate(options, start=1):
        print(
            f"  {index:>3}  {option.cloud:<16} {option.display_name:<24} "
            f"{option.shade_instance_type:<16} {option.gpu_type:<12} "
            f"${option.hourly_price_dollars:>6.2f}",
            file=sys.stderr,
        )


def build_training_command(
    *,
    target: TrainingTarget,
    config_argument: str,
    csv_argument: str,
    output_root: str,
    run_name: str | None,
    epochs: int | None,
    batch_size: int | None,
    limit_rows: int | None,
    limit_samples: int | None,
    sequence_length: int | None,
    window_stride: int | None,
    max_trials: int | None,
    trial_output_root: Path | str | None,
    continue_on_error: bool,
    no_plots: bool,
    dry_run: bool,
    device: str,
    num_gpus: int,
    remote_workdir: str,
    remote_env: Mapping[str, str],
    extra_training_args: Sequence[str],
) -> str:
    launcher_args = [
        "uv",
        "run",
        "--python",
        REMOTE_PYTHON_VERSION,
        "--no-sync",
        "python",
        "-m",
    ]
    if num_gpus > 1:
        args = [
            *launcher_args,
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            str(num_gpus),
            "--module",
            target.entrypoint,
        ]
    else:
        args = [*launcher_args, target.entrypoint]
    args.extend(
        [
            "--config",
            config_argument,
            "--csv",
            csv_argument,
            "--output-root",
            output_root,
            "--device",
            device,
        ]
    )
    if run_name:
        args.extend([target.run_name_argument, run_name])
    if epochs is not None:
        args.extend(["--epochs", str(epochs)])
    if batch_size is not None:
        args.extend(["--batch-size", str(batch_size)])
    if target.limit_argument == "--limit-rows" and limit_rows is not None:
        args.extend([target.limit_argument, str(limit_rows)])
    if target.limit_argument == "--limit-samples" and limit_samples is not None:
        args.extend([target.limit_argument, str(limit_samples)])
    if target.supports_sequence_args and sequence_length is not None:
        args.extend(["--sequence-length", str(sequence_length)])
    if target.supports_sequence_args and window_stride is not None:
        args.extend(["--window-stride", str(window_stride)])
    if target.supports_grid_args and max_trials is not None:
        args.extend(["--max-trials", str(max_trials)])
    if target.supports_grid_args and trial_output_root is not None:
        args.extend(["--trial-output-root", normalize_remote_relative_path(trial_output_root)])
    if target.supports_grid_args and continue_on_error:
        args.append("--continue-on-error")
    if no_plots:
        args.append("--no-plots")
    if dry_run:
        args.append("--dry-run")
    args.extend(str(argument) for argument in extra_training_args)

    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(remote_env.items()))
    command = " ".join(shlex.quote(argument) for argument in args)
    if env_prefix:
        command = f"{env_prefix} {command}"
    return f"{UV_ENV} && cd {shlex.quote(remote_workdir)} && {command}"


def build_upload_specs(
    project_root: Path,
    *,
    remote_workdir: str,
    config_path: Path,
    csv_path: Path,
    extra_upload_paths: Sequence[Path | str],
    include_csv_upload: bool,
    include_gdrive_helper: bool,
    validate_paths: bool,
) -> list[UploadSpec]:
    upload_paths: list[Path] = [
        project_root / "pyproject.toml",
        project_root / "hlgam.env",
        project_root / "uv.lock",
        project_root / "hlgam",
        project_root / "training",
        config_path,
        *(resolve_local_path(path, project_root) for path in extra_upload_paths),
    ]
    if include_csv_upload:
        upload_paths.append(csv_path)
    if include_gdrive_helper:
        upload_paths.append(project_root / GDRIVE_HELPER_PATH)
    specs: list[UploadSpec] = []
    for local_path in upload_paths:
        if not local_path.exists():
            if validate_paths:
                raise FileNotFoundError(f"Shadeform upload path does not exist: {local_path}")
            continue
        if is_covered_by_directory(local_path, specs):
            continue
        remote_argument = remote_argument_for_local_path(local_path, project_root, remote_workdir)
        specs.append(UploadSpec(local_path=local_path, remote_path=remote_join(remote_workdir, remote_argument)))
    return dedupe_upload_specs(specs)


def target_for_model_family(model_family: str, training_job: str = "train") -> TrainingTarget:
    try:
        return TRAINING_TARGETS[(model_family, training_job)]
    except KeyError as exc:
        families = ", ".join(sorted({family for family, _ in TRAINING_TARGETS}))
        jobs = ", ".join(sorted({job for _, job in TRAINING_TARGETS}))
        raise ValueError(f"model_family must be one of: {families}; training_job must be one of: {jobs}") from exc


def normalize_num_gpus(num_gpus: int | None) -> int:
    resolved = 1 if num_gpus is None else int(num_gpus)
    if resolved < 1:
        raise ValueError("num_gpus must be greater than or equal to 1")
    return resolved


def resolve_local_path(path: Path | str, project_root: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = project_root / resolved
    return resolved.resolve()


def remote_argument_for_local_path(local_path: Path, project_root: Path, remote_workdir: str) -> str:
    try:
        return local_path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        digest = hashlib.sha1(str(local_path.resolve()).encode("utf-8")).hexdigest()[:10]
        return f".shadeform_uploads/{digest}_{local_path.name}"


def remote_join(remote_workdir: str, relative_path: str) -> str:
    return f"{remote_workdir.rstrip('/')}/{relative_path.strip('/')}"


def normalize_remote_relative_path(path: Path | str) -> str:
    normalized = str(path).replace("\\", "/").strip()
    if not normalized:
        raise ValueError("remote path cannot be empty")
    if normalized.startswith("/"):
        raise ValueError("remote training paths must be relative to the remote project workdir")
    return normalized.strip("/")


def unique_paths(paths: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        normalized = normalize_remote_relative_path(path)
        if normalized not in seen:
            unique.append(normalized)
            seen.add(normalized)
    return tuple(unique)


def is_covered_by_directory(local_path: Path, specs: Sequence[UploadSpec]) -> bool:
    for spec in specs:
        if not spec.local_path.is_dir():
            continue
        try:
            local_path.resolve().relative_to(spec.local_path.resolve())
        except ValueError:
            continue
        return True
    return False


def dedupe_upload_specs(specs: Sequence[UploadSpec]) -> list[UploadSpec]:
    by_remote_path: dict[str, UploadSpec] = {}
    for spec in specs:
        by_remote_path.setdefault(spec.remote_path, spec)
    return list(by_remote_path.values())


def filter_options(
    options: Sequence[AvailabilityOption],
    *,
    cloud: str | None,
    region: str | None,
    shade_instance_type: str | None,
) -> list[AvailabilityOption]:
    filtered = [
        option
        for option in options
        if (cloud is None or option.cloud == cloud)
        and (region is None or option.region == region)
        and (shade_instance_type is None or option.shade_instance_type == shade_instance_type)
    ]
    filtered.sort(key=lambda option: option.hourly_price_cents)
    return filtered


def direct_provider_option(
    *,
    cloud: str,
    region: str,
    shade_instance_type: str,
    gpu_type: str,
    num_gpus: int,
) -> AvailabilityOption:
    return AvailabilityOption(
        cloud=cloud,
        region=region,
        display_name=region,
        shade_instance_type=shade_instance_type,
        hourly_price_cents=0,
        num_gpus=num_gpus,
        gpu_type=gpu_type,
        vram_per_gpu_gb=0,
        memory_gb=0,
        storage_gb=0,
        vcpus=0,
    )


def provider_to_dict(provider: AvailabilityOption) -> dict[str, Any]:
    return {
        "cloud": provider.cloud,
        "region": provider.region,
        "display_name": provider.display_name,
        "shade_instance_type": provider.shade_instance_type,
        "hourly_price_cents": provider.hourly_price_cents,
        "num_gpus": provider.num_gpus,
        "gpu_type": provider.gpu_type,
        "vram_per_gpu_gb": provider.vram_per_gpu_gb,
        "memory_gb": provider.memory_gb,
        "storage_gb": provider.storage_gb,
        "vcpus": provider.vcpus,
    }


def plan_to_json(plan: RemoteTrainingPlan) -> str:
    return json.dumps(plan.to_dict(), indent=2, sort_keys=True)


def build_gdrive_plan(
    *,
    input_csv_file_id: str,
    input_csv_remote_path: str,
    results_folder_id: str,
    results_subfolder: str,
    service_account_remote_path: str,
) -> GoogleDrivePlan:
    return GoogleDrivePlan(
        service_account_remote_path=service_account_remote_path,
        input_csv_file_id=input_csv_file_id,
        input_csv_remote_path=input_csv_remote_path,
        results_folder_id=results_folder_id,
        results_subfolder=results_subfolder,
    )


def build_pre_training_commands(gdrive_plan: GoogleDrivePlan, remote_workdir: str) -> tuple[str, ...]:
    return (
        build_gdrive_download_file_command(
            remote_workdir=remote_workdir,
            service_account_remote_path=gdrive_plan.service_account_remote_path,
            file_id=gdrive_plan.input_csv_file_id,
            destination_path=gdrive_plan.input_csv_remote_path,
        ),
    )


def build_post_training_commands(
    gdrive_plan: GoogleDrivePlan,
    remote_workdir: str,
    *,
    return_paths: Sequence[str],
) -> tuple[str, ...]:
    return (
        build_gdrive_upload_command(
            remote_workdir=remote_workdir,
            service_account_remote_path=gdrive_plan.service_account_remote_path,
            parent_folder_id=gdrive_plan.results_folder_id,
            subfolder_name=gdrive_plan.results_subfolder,
            return_paths=return_paths,
        ),
    )


def build_gdrive_download_file_command(
    *,
    remote_workdir: str,
    service_account_remote_path: str,
    file_id: str,
    destination_path: str,
) -> str:
    args = [
        "uv",
        "run",
        "--python",
        REMOTE_PYTHON_VERSION,
        "--with",
        "google-api-python-client",
        "python",
        "shadeform/gdrive.py",
        "download-file",
        "--sa-key",
        service_account_remote_path,
        "--file-id",
        file_id,
        "--dest",
        destination_path,
    ]
    command = " ".join(shlex.quote(argument) for argument in args)
    return f"{UV_ENV} && cd {shlex.quote(remote_workdir)} && {command}"


def build_gdrive_upload_command(
    *,
    remote_workdir: str,
    service_account_remote_path: str,
    parent_folder_id: str,
    subfolder_name: str,
    return_paths: Sequence[str],
) -> str:
    args = [
        "uv",
        "run",
        "--python",
        REMOTE_PYTHON_VERSION,
        "--with",
        "google-api-python-client",
        "python",
        "shadeform/gdrive.py",
        "upload",
        "--sa-key",
        service_account_remote_path,
        "--parent",
        parent_folder_id,
        "--subfolder",
        subfolder_name,
        "--root",
        remote_workdir,
    ]
    for return_path in return_paths:
        args.extend(["--path", return_path])
    command = " ".join(shlex.quote(argument) for argument in args)
    return f"{UV_ENV} && cd {shlex.quote(remote_workdir)} && {command}"


def build_gdrive_download_command(
    *,
    service_account_key_path: Path | str,
    parent_folder_id: str,
    subfolder_name: str,
    destination_root: Path | str,
    return_paths: Sequence[str],
) -> list[str]:
    args = [
        "uv",
        "run",
        "--with",
        "google-api-python-client",
        "python",
        GDRIVE_HELPER_PATH.as_posix(),
        "download",
        "--sa-key",
        str(service_account_key_path),
        "--parent",
        parent_folder_id,
        "--subfolder",
        subfolder_name,
        "--dest",
        str(destination_root),
    ]
    for return_path in return_paths:
        args.extend(["--path", normalize_remote_relative_path(return_path)])
    return args


def planned_drive_upload_paths(plan: RemoteTrainingPlan) -> list[str]:
    return list(plan.return_paths)


def require_nonempty_text(value: str | None, field_name: str, guidance: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{field_name} is required. {guidance}")
    return value.strip()
