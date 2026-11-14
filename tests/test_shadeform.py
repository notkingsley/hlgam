from __future__ import annotations

import json
import subprocess
from pathlib import Path

from shadeform.cli import main
from shadeform.config import ShadeformSettings, load_settings, parse_env_assignments
from shadeform.runner import (
    PROJECT_ROOT,
    build_gdrive_download_command,
    build_remote_training_plan,
    download_return_paths_from_gdrive,
)


def test_static_pose_plan_uploads_project_without_csv_or_image_tree(tmp_path: Path) -> None:
    project_root = write_minimal_project(tmp_path)

    plan = build_remote_training_plan(
        model_family="static_pose",
        project_root=project_root,
        gdrive_csv_file_id="static-drive-csv",
        gdrive_results_folder_id="drive-results-folder",
        epochs=3,
        batch_size=4,
        limit_rows=10,
        run_name="pilot",
    )

    upload_remotes = {upload.remote_path for upload in plan.upload_paths}
    assert f"{plan.remote_workdir}/hlgam" in upload_remotes
    assert f"{plan.remote_workdir}/training" in upload_remotes
    assert f"{plan.remote_workdir}/shadeform/gdrive.py" in upload_remotes
    assert f"{plan.remote_workdir}/data/static_poses/static_pose_frames.csv" not in upload_remotes
    assert not any("/images" in remote_path for remote_path in upload_remotes)
    assert "training.jobs.train_static_pose" in plan.training_command
    assert "--limit-rows 10" in plan.training_command
    assert "--extra ml" not in " ".join(plan.setup_commands)
    assert "--torch-backend=auto" in " ".join(plan.setup_commands)
    assert "download-file" in plan.pre_training_commands[0]
    assert "shadeform/gdrive.py upload" in plan.post_training_commands[0]
    assert plan.return_paths == ("runs/static_pose",)


def test_motion_plan_adds_motion_specific_training_args(tmp_path: Path) -> None:
    project_root = write_minimal_project(tmp_path)

    plan = build_remote_training_plan(
        model_family="motion",
        project_root=project_root,
        gdrive_csv_file_id="motion-drive-csv",
        gdrive_results_folder_id="drive-results-folder",
        limit_samples=8,
        sequence_length=16,
        window_stride=2,
        no_plots=True,
        dry_run=True,
        remote_env={"WANDB_MODE": "offline"},
        return_paths=("logs",),
    )

    assert "training.jobs.train_motion" in plan.training_command
    assert "--limit-samples 8" in plan.training_command
    assert "--sequence-length 16" in plan.training_command
    assert "--window-stride 2" in plan.training_command
    assert "--no-plots" in plan.training_command
    assert "--dry-run" in plan.training_command
    assert "WANDB_MODE=offline" in plan.training_command
    assert plan.return_paths == ("runs/motion", "logs")


def test_motion_loss_grid_plan_uses_grid_entrypoint_and_outputs(tmp_path: Path) -> None:
    project_root = write_minimal_project(tmp_path)

    plan = build_remote_training_plan(
        model_family="motion",
        training_job="loss_grid",
        project_root=project_root,
        gdrive_csv_file_id="motion-drive-csv",
        gdrive_results_folder_id="drive-results-folder",
        run_name="loss-sweep",
        epochs=2,
        batch_size=4,
        limit_samples=8,
        sequence_length=16,
        window_stride=2,
        max_trials=3,
        trial_output_root="runs/motion_loss_grid_trials",
        continue_on_error=True,
        no_plots=True,
        dry_run=True,
    )

    assert plan.model_family == "motion"
    assert plan.training_job == "loss_grid"
    assert plan.instance_name.startswith("hlgam-motion-loss-grid-")
    assert plan.entrypoint == "training.jobs.grid_search_motion_loss"
    assert "training.jobs.grid_search_motion_loss" in plan.training_command
    assert "--config training/configs/motion_loss_grid.yaml" in plan.training_command
    assert "--csv data/motion_sequences/motion_sequence_frames.csv" in plan.training_command
    assert "--output-root runs/motion_loss_grid" in plan.training_command
    assert "--search-name loss-sweep" in plan.training_command
    assert "--epochs 2" in plan.training_command
    assert "--batch-size 4" in plan.training_command
    assert "--limit-samples 8" in plan.training_command
    assert "--sequence-length 16" in plan.training_command
    assert "--window-stride 2" in plan.training_command
    assert "--max-trials 3" in plan.training_command
    assert "--trial-output-root runs/motion_loss_grid_trials" in plan.training_command
    assert "--continue-on-error" in plan.training_command
    assert "--no-plots" in plan.training_command
    assert "--dry-run" in plan.training_command
    assert plan.return_paths == ("runs/motion_loss_grid",)
    assert plan.gdrive.input_csv_file_id == "motion-drive-csv"


def test_gdrive_csv_plan_downloads_remote_csv_instead_of_uploading_local_csv(tmp_path: Path) -> None:
    project_root = write_minimal_project(tmp_path)
    local_csv = project_root / "data" / "motion_sequences" / "motion_sequence_frames.csv"
    local_csv.unlink()

    plan = build_remote_training_plan(
        model_family="motion",
        project_root=project_root,
        gdrive_csv_file_id="drive-csv-file-id",
        gdrive_results_folder_id="drive-results-folder-id",
        no_plots=True,
    )

    upload_remotes = {upload.remote_path for upload in plan.upload_paths}
    remote_csv = f"{plan.remote_workdir}/data/motion_sequences/motion_sequence_frames.csv"
    assert remote_csv not in upload_remotes
    assert f"{plan.remote_workdir}/shadeform/gdrive.py" in upload_remotes
    assert plan.gdrive is not None
    assert plan.gdrive.input_csv_file_id == "drive-csv-file-id"
    assert plan.gdrive.results_folder_id == "drive-results-folder-id"
    assert plan.gdrive.input_csv_remote_path == remote_csv
    assert len(plan.pre_training_commands) == 1
    assert "download-file" in plan.pre_training_commands[0]
    assert "--file-id drive-csv-file-id" in plan.pre_training_commands[0]
    assert f"--dest {remote_csv}" in plan.pre_training_commands[0]
    assert "--csv data/motion_sequences/motion_sequence_frames.csv" in plan.training_command


def test_gdrive_results_plan_uploads_default_and_extra_returns(tmp_path: Path) -> None:
    project_root = write_minimal_project(tmp_path)

    plan = build_remote_training_plan(
        model_family="static_pose",
        project_root=project_root,
        gdrive_csv_file_id="drive-csv-file-id",
        gdrive_results_folder_id="drive-results-folder-id",
        return_paths=("logs",),
        instance_name="static-gdrive-run",
    )

    upload_remotes = {upload.remote_path for upload in plan.upload_paths}
    assert f"{plan.remote_workdir}/shadeform/gdrive.py" in upload_remotes
    assert plan.gdrive is not None
    assert plan.gdrive.results_folder_id == "drive-results-folder-id"
    assert plan.instance_name.startswith("static-gdrive-run-")
    assert plan.gdrive.results_subfolder == plan.instance_name
    assert plan.return_paths == ("runs/static_pose", "logs")
    assert len(plan.post_training_commands) == 1
    assert "shadeform/gdrive.py upload" in plan.post_training_commands[0]
    assert "--parent drive-results-folder-id" in plan.post_training_commands[0]
    assert f"--subfolder {plan.instance_name}" in plan.post_training_commands[0]
    assert "--path runs/static_pose" in plan.post_training_commands[0]
    assert "--path logs" in plan.post_training_commands[0]


def test_remote_plan_pins_uv_python_to_312(tmp_path: Path) -> None:
    project_root = write_minimal_project(tmp_path)

    plan = build_remote_training_plan(
        model_family="motion",
        project_root=project_root,
        gdrive_csv_file_id="motion-drive-csv",
        gdrive_results_folder_id="drive-results-folder-id",
    )

    assert plan.num_gpus == 1
    assert "uv python install 3.12" in plan.setup_commands[1]
    assert "uv sync --python 3.12 --no-dev" in plan.setup_commands[2]
    assert "uv run --python 3.12 --no-sync python -m training.jobs.train_motion" in plan.training_command
    assert (
        "uv run --python 3.12 --with google-api-python-client python shadeform/gdrive.py download-file"
        in plan.pre_training_commands[0]
    )
    assert (
        "uv run --python 3.12 --with google-api-python-client python shadeform/gdrive.py upload"
        in plan.post_training_commands[0]
    )


def test_multi_gpu_plan_uses_torch_distributed_run(tmp_path: Path) -> None:
    project_root = write_minimal_project(tmp_path)

    plan = build_remote_training_plan(
        model_family="motion",
        project_root=project_root,
        gdrive_csv_file_id="motion-drive-csv",
        gdrive_results_folder_id="drive-results-folder-id",
        num_gpus=4,
    )

    assert plan.num_gpus == 4
    assert (
        "uv run --python 3.12 --no-sync python -m torch.distributed.run "
        "--standalone --nproc-per-node 4 --module training.jobs.train_motion"
        in plan.training_command
    )
    assert "--device auto" in plan.training_command


def test_gdrive_download_command_targets_run_subfolder(tmp_path: Path) -> None:
    service_account_key_path = tmp_path / "service-account.json"
    destination_root = tmp_path / "shadeform_results" / "static-gdrive-run"

    command = build_gdrive_download_command(
        service_account_key_path=service_account_key_path,
        parent_folder_id="drive-results-folder-id",
        subfolder_name="static-gdrive-run",
        destination_root=destination_root,
        return_paths=("runs/static_pose", "logs"),
    )

    assert command[:7] == [
        "uv",
        "run",
        "--with",
        "google-api-python-client",
        "python",
        "shadeform/gdrive.py",
        "download",
    ]
    assert command[command.index("--sa-key") + 1] == str(service_account_key_path)
    assert command[command.index("--parent") + 1] == "drive-results-folder-id"
    assert command[command.index("--subfolder") + 1] == "static-gdrive-run"
    assert command[command.index("--dest") + 1] == str(destination_root)
    assert command[-4:] == ["--path", "runs/static_pose", "--path", "logs"]


def test_download_return_paths_from_gdrive_runs_local_download(tmp_path: Path, monkeypatch) -> None:
    project_root = write_minimal_project(tmp_path)
    ssh_key_path = tmp_path / "id_ed25519"
    ssh_key_path.write_text("not-a-real-key", encoding="utf-8")
    service_account_key_path = tmp_path / "service-account.json"
    service_account_key_path.write_text("{}", encoding="utf-8")
    settings = ShadeformSettings(
        api_key="test-key",
        ssh_key_path=ssh_key_path,
        results_dir=tmp_path / "shadeform_results",
        gdrive_service_account_key_path=service_account_key_path,
        gdrive_results_folder_id="drive-results-folder-id",
        gdrive_motion_csv_file_id="motion-drive-csv",
    )
    plan = build_remote_training_plan(
        model_family="motion",
        project_root=project_root,
        gdrive_csv_file_id="motion-drive-csv",
        gdrive_results_folder_id="drive-results-folder-id",
        return_paths=("logs",),
        instance_name="motion-gdrive-run",
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[list[str]]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("shadeform.runner.subprocess.run", fake_run)

    downloaded_paths = download_return_paths_from_gdrive(
        settings,
        plan,
        service_account_key_path=service_account_key_path,
    )

    destination_root = settings.results_dir / plan.instance_name
    assert downloaded_paths == [
        str(destination_root / "runs/motion"),
        str(destination_root / "logs"),
    ]
    assert calls == [
        (
            build_gdrive_download_command(
                service_account_key_path=service_account_key_path,
                parent_folder_id="drive-results-folder-id",
                subfolder_name=plan.instance_name,
                destination_root=destination_root,
                return_paths=("runs/motion", "logs"),
            ),
            {"cwd": PROJECT_ROOT, "text": True, "check": False},
        )
    ]


def test_gdrive_download_parser_accepts_selected_return_paths(tmp_path: Path) -> None:
    from shadeform.gdrive import build_parser

    args = build_parser().parse_args(
        [
            "download",
            "--sa-key",
            str(tmp_path / "service-account.json"),
            "--parent",
            "drive-results-folder-id",
            "--subfolder",
            "motion-gdrive-run",
            "--dest",
            str(tmp_path / "shadeform_results" / "motion-gdrive-run"),
            "--path",
            "runs/motion",
            "--path",
            "logs",
        ]
    )

    assert args.command == "download"
    assert args.paths == ["runs/motion", "logs"]


def test_build_plan_requires_google_drive_ids(tmp_path: Path) -> None:
    project_root = write_minimal_project(tmp_path)

    try:
        build_remote_training_plan(model_family="motion", project_root=project_root)
    except ValueError as exc:
        assert "gdrive_csv_file_id" in str(exc)
    else:
        raise AssertionError("build_remote_training_plan accepted missing Google Drive IDs")


def test_build_plan_requires_google_drive_results_folder(tmp_path: Path) -> None:
    project_root = write_minimal_project(tmp_path)

    try:
        build_remote_training_plan(
            model_family="motion",
            project_root=project_root,
            gdrive_csv_file_id="drive-csv-file-id",
        )
    except ValueError as exc:
        assert "gdrive_results_folder_id" in str(exc)
    else:
        raise AssertionError("build_remote_training_plan accepted a missing Drive results folder")


def test_load_settings_reads_google_drive_values(tmp_path: Path) -> None:
    ssh_key_path = tmp_path / "id_ed25519"
    ssh_key_path.write_text("not-a-real-key", encoding="utf-8")
    service_account_path = tmp_path / "service-account.json"
    service_account_path.write_text("{}", encoding="utf-8")
    env_file = tmp_path / "shadeform.env"
    env_file.write_text(
        "\n".join(
            [
                "SHADEFORM_API_KEY=test-key",
                f"SHADEFORM_SSH_KEY_PATH={ssh_key_path}",
                f"SHADEFORM_GDRIVE_SERVICE_ACCOUNT_KEY_PATH={service_account_path}",
                "SHADEFORM_GDRIVE_RESULTS_FOLDER_ID=results-folder",
                "SHADEFORM_GDRIVE_CSV_FILE_ID=generic-csv",
                "SHADEFORM_GDRIVE_STATIC_POSE_CSV_FILE_ID=static-csv",
                "SHADEFORM_GDRIVE_MOTION_CSV_FILE_ID=motion-csv",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(env_file)

    assert settings.gdrive_service_account_key_path == service_account_path
    assert settings.gdrive_results_folder_id == "results-folder"
    assert settings.gdrive_input_csv_file_id_for("static_pose") == "static-csv"
    assert settings.gdrive_input_csv_file_id_for("motion") == "motion-csv"
    assert settings.gdrive_input_csv_file_id_for("other") == "generic-csv"


def test_cli_plan_prints_json_without_shadeform_credentials(tmp_path: Path, capsys) -> None:
    project_root = write_minimal_project(tmp_path)

    exit_code = main(
        [
            "train",
            "--model-family",
            "motion",
            "--project-root",
            str(project_root),
            "--plan",
            "--gdrive-csv-file-id",
            "motion-drive-csv",
            "--gdrive-results-folder-id",
            "drive-results-folder",
            "--epochs",
            "1",
            "--env",
            "TOKEN=abc123",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["model_family"] == "motion"
    assert payload["remote_env"] == {"TOKEN": "abc123"}
    assert payload["gdrive"]["input_csv_file_id"] == "motion-drive-csv"
    assert payload["gdrive"]["results_folder_id"] == "drive-results-folder"
    assert "training.jobs.train_motion" in payload["training_command"]


def test_cli_motion_loss_grid_plan_prints_grid_command(tmp_path: Path, capsys) -> None:
    project_root = write_minimal_project(tmp_path)

    exit_code = main(
        [
            "train",
            "--model-family",
            "motion",
            "--job",
            "loss_grid",
            "--project-root",
            str(project_root),
            "--plan",
            "--gdrive-csv-file-id",
            "motion-drive-csv",
            "--gdrive-results-folder-id",
            "drive-results-folder",
            "--epochs",
            "1",
            "--max-trials",
            "2",
            "--no-plots",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["model_family"] == "motion"
    assert payload["training_job"] == "loss_grid"
    assert payload["entrypoint"] == "training.jobs.grid_search_motion_loss"
    assert "--max-trials 2" in payload["training_command"]
    assert payload["return_paths"] == ["runs/motion_loss_grid"]


def test_cli_plan_uses_num_gpus_for_distributed_command(tmp_path: Path, capsys) -> None:
    project_root = write_minimal_project(tmp_path)

    exit_code = main(
        [
            "train",
            "--model-family",
            "static_pose",
            "--project-root",
            str(project_root),
            "--plan",
            "--num-gpus",
            "2",
            "--gdrive-csv-file-id",
            "static-drive-csv",
            "--gdrive-results-folder-id",
            "drive-results-folder",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["num_gpus"] == 2
    assert "torch.distributed.run" in payload["training_command"]
    assert "--nproc-per-node 2" in payload["training_command"]
    assert "--module training.jobs.train_static_pose" in payload["training_command"]


def test_cli_plan_reads_google_drive_ids_from_env_file(tmp_path: Path, capsys) -> None:
    project_root = write_minimal_project(tmp_path)
    env_file = tmp_path / "shadeform.env"
    env_file.write_text(
        "SHADEFORM_GDRIVE_MOTION_CSV_FILE_ID=motion-env-csv\n"
        "SHADEFORM_GDRIVE_RESULTS_FOLDER_ID=env-results-folder\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "train",
            "--model-family",
            "motion",
            "--project-root",
            str(project_root),
            "--env-file",
            str(env_file),
            "--plan",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["gdrive"]["input_csv_file_id"] == "motion-env-csv"
    assert payload["gdrive"]["results_folder_id"] == "env-results-folder"


def test_cli_plan_includes_google_drive_commands(tmp_path: Path, capsys) -> None:
    project_root = write_minimal_project(tmp_path)

    exit_code = main(
        [
            "train",
            "--model-family",
            "static_pose",
            "--project-root",
            str(project_root),
            "--plan",
            "--gdrive-csv-file-id",
            "input-csv-file",
            "--gdrive-results-folder-id",
            "results-folder",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["gdrive"]["input_csv_file_id"] == "input-csv-file"
    assert payload["gdrive"]["results_folder_id"] == "results-folder"
    assert "download-file" in payload["pre_training_commands"][0]
    assert "upload" in payload["post_training_commands"][0]


def test_parse_env_assignments_rejects_missing_separator() -> None:
    try:
        parse_env_assignments(["TOKEN"])
    except ValueError as exc:
        assert "KEY=VALUE" in str(exc)
    else:
        raise AssertionError("parse_env_assignments accepted an invalid assignment")


def write_minimal_project(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    (project_root / "hlgam").mkdir(parents=True)
    (project_root / "shadeform").mkdir(parents=True)
    (project_root / "training" / "configs").mkdir(parents=True)
    (project_root / "training" / "label_schemas").mkdir(parents=True)
    (project_root / "data" / "static_poses").mkdir(parents=True)
    (project_root / "data" / "motion_sequences").mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname = 'gesture-app'\n", encoding="utf-8")
    (project_root / "README.md").write_text("# gesture app\n", encoding="utf-8")
    (project_root / "hlgam.env").write_text("\n", encoding="utf-8")
    (project_root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (project_root / "hlgam" / "__init__.py").write_text("", encoding="utf-8")
    (project_root / "shadeform" / "gdrive.py").write_text("", encoding="utf-8")
    (project_root / "training" / "__init__.py").write_text("", encoding="utf-8")
    (project_root / "training" / "configs" / "static_pose_transformer.yaml").write_text("run: {}\n", encoding="utf-8")
    (project_root / "training" / "configs" / "motion_transformer.yaml").write_text("run: {}\n", encoding="utf-8")
    (project_root / "training" / "configs" / "static_pose_loss_grid.yaml").write_text(
        "base_config: training/configs/static_pose_transformer.yaml\n",
        encoding="utf-8",
    )
    (project_root / "training" / "configs" / "motion_loss_grid.yaml").write_text(
        "base_config: training/configs/motion_transformer.yaml\n",
        encoding="utf-8",
    )
    (project_root / "data" / "static_poses" / "static_pose_frames.csv").write_text("schema_version\n", encoding="utf-8")
    (project_root / "data" / "motion_sequences" / "motion_sequence_frames.csv").write_text(
        "schema_version\n",
        encoding="utf-8",
    )
    return project_root
