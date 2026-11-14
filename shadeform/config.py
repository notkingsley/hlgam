"""Configuration loading for Shadeform GPU training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


DEFAULT_ENV_FILE = Path("shadeform.env")
DEFAULT_RESULTS_DIR = Path("shadeform_results")
DEFAULT_SSH_USER = "shadeform"


@dataclass(frozen=True)
class ShadeformSettings:
    api_key: str
    ssh_key_path: Path
    ssh_user: str = DEFAULT_SSH_USER
    results_dir: Path = DEFAULT_RESULTS_DIR
    default_gpu_type: str = "A100"
    default_num_gpus: int = 1
    default_cloud: str | None = None
    default_region: str | None = None
    default_shade_instance_type: str | None = None
    gdrive_service_account_key_path: Path | None = None
    gdrive_results_folder_id: str | None = None
    gdrive_csv_file_id: str | None = None
    gdrive_static_pose_csv_file_id: str | None = None
    gdrive_motion_csv_file_id: str | None = None

    def validate(self) -> None:
        if not self.api_key:
            raise ValueError("SHADEFORM_API_KEY is required")
        if not self.ssh_key_path:
            raise ValueError("SHADEFORM_SSH_KEY_PATH is required")
        if not self.ssh_key_path.exists():
            raise FileNotFoundError(f"SHADEFORM_SSH_KEY_PATH does not exist: {self.ssh_key_path}")

    def gdrive_input_csv_file_id_for(self, model_family: str) -> str | None:
        if model_family == "static_pose" and self.gdrive_static_pose_csv_file_id:
            return self.gdrive_static_pose_csv_file_id
        if model_family == "motion" and self.gdrive_motion_csv_file_id:
            return self.gdrive_motion_csv_file_id
        return self.gdrive_csv_file_id

    def require_gdrive_service_account_key_path(self) -> Path:
        if self.gdrive_service_account_key_path is None:
            raise ValueError(
                "SHADEFORM_GDRIVE_SERVICE_ACCOUNT_KEY_PATH is required for Google Drive transfers"
            )
        if not self.gdrive_service_account_key_path.exists():
            raise FileNotFoundError(
                "SHADEFORM_GDRIVE_SERVICE_ACCOUNT_KEY_PATH does not exist: "
                f"{self.gdrive_service_account_key_path}"
            )
        return self.gdrive_service_account_key_path


def load_settings(
    env_file: Path | str | None = DEFAULT_ENV_FILE,
    *,
    environ: Mapping[str, str] | None = None,
    validate: bool = True,
) -> ShadeformSettings:
    values = load_env_values(env_file, environ=environ)
    settings = ShadeformSettings(
        api_key=require_value(values, "SHADEFORM_API_KEY"),
        ssh_key_path=Path(require_value(values, "SHADEFORM_SSH_KEY_PATH")).expanduser(),
        ssh_user=optional_value(values, "SHADEFORM_SSH_USER") or DEFAULT_SSH_USER,
        results_dir=Path(optional_value(values, "SHADEFORM_RESULTS_DIR") or DEFAULT_RESULTS_DIR).expanduser(),
        default_gpu_type=optional_value(values, "SHADEFORM_GPU_TYPE") or "A100",
        default_num_gpus=int(optional_value(values, "SHADEFORM_NUM_GPUS") or 1),
        default_cloud=optional_value(values, "SHADEFORM_CLOUD"),
        default_region=optional_value(values, "SHADEFORM_REGION"),
        default_shade_instance_type=optional_value(values, "SHADEFORM_SHADE_INSTANCE_TYPE"),
        gdrive_service_account_key_path=optional_path_value(
            values,
            "SHADEFORM_GDRIVE_SERVICE_ACCOUNT_KEY_PATH",
            "GDRIVE_SA_KEY_PATH",
        ),
        gdrive_results_folder_id=optional_value(
            values,
            "SHADEFORM_GDRIVE_RESULTS_FOLDER_ID",
            "GDRIVE_RESULTS_FOLDER_ID",
        ),
        gdrive_csv_file_id=optional_value(values, "SHADEFORM_GDRIVE_CSV_FILE_ID"),
        gdrive_static_pose_csv_file_id=optional_value(
            values,
            "SHADEFORM_GDRIVE_STATIC_POSE_CSV_FILE_ID",
        ),
        gdrive_motion_csv_file_id=optional_value(values, "SHADEFORM_GDRIVE_MOTION_CSV_FILE_ID"),
    )
    if validate:
        settings.validate()
    return settings


def load_env_values(
    env_file: Path | str | None = DEFAULT_ENV_FILE,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    values: dict[str, str] = {}
    if env_file is not None:
        path = Path(env_file).expanduser()
        if path.exists():
            values.update({key: value for key, value in dotenv_values(path).items() if value is not None})
    values.update(dict(os.environ if environ is None else environ))
    return {key: value for key, value in values.items() if value is not None}


def parse_env_assignments(assignments: list[str] | tuple[str, ...] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for assignment in assignments or ():
        key, separator, value = assignment.partition("=")
        key = key.strip()
        if not key or not separator:
            raise ValueError(f"remote env assignment must be KEY=VALUE: {assignment!r}")
        parsed[key] = value
    return parsed


def require_value(values: Mapping[str, str], key: str) -> str:
    value = optional_value(values, key)
    if value is None:
        raise ValueError(f"{key} is required")
    return value


def optional_path_value(values: Mapping[str, str], *keys: str) -> Path | None:
    value = optional_value(values, *keys)
    if value is None:
        return None
    return Path(value).expanduser()


def optional_value(values: Mapping[str, str], *keys: str) -> str | None:
    for key in keys:
        value = values.get(key)
        if value is None:
            continue
        stripped = value.strip()
        if stripped:
            return stripped
    return None
