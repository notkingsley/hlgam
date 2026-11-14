from __future__ import annotations

import csv
from pathlib import Path

import pytest

from training.capture_motion_sequences import FIELDNAMES, MOTION_LABELS

pytest.importorskip("torch")
pytest.importorskip("matplotlib")


def test_motion_window_dataset_parses_capture_csv(tmp_path: Path) -> None:
    from training.datasets.motion_window_dataset import MotionWindowDataset, motion_window_collate

    csv_path = write_motion_csv(tmp_path)
    dataset = MotionWindowDataset(csv_path, sequence_length=4, window_stride=2, min_interval_overlap=0.5)
    positive_sample = next(sample for sample in dataset if sample["labels"].sum().item() > 0)
    batch = motion_window_collate([dataset[0], positive_sample])
    profile = dataset.profile()

    assert len(dataset) == 12
    assert positive_sample["landmarks"].shape == (4, 21, 3)
    assert positive_sample["global_features"].shape == (4, 22)
    assert positive_sample["labels"].shape == (len(MOTION_LABELS),)
    assert positive_sample["phase_targets"].shape == (len(MOTION_LABELS),)
    assert positive_sample["phase_mask"].shape == (len(MOTION_LABELS),)
    assert positive_sample["landmark_mask"].shape == (4, 21)
    assert batch["landmarks"].shape == (2, 4, 21, 3)
    assert batch["labels"].shape == (2, len(MOTION_LABELS))
    assert profile.total_samples == 3
    assert profile.total_frames == 18
    assert profile.label_counts["DRAG_MOVE"] > 0


def test_motion_model_and_loss_accept_dataset_batch(tmp_path: Path) -> None:
    from hlgam.ml.models.motion_transformer import MotionGestureTransformer, MotionGestureTransformerConfig
    from training.common.losses import MotionGestureLoss, MotionGestureLossConfig
    from training.datasets.motion_window_dataset import MotionWindowDataset, motion_window_collate

    csv_path = write_motion_csv(tmp_path)
    dataset = MotionWindowDataset(csv_path, sequence_length=4, window_stride=2, min_interval_overlap=0.5)
    batch = motion_window_collate([dataset[0], dataset[1]])
    model = MotionGestureTransformer(
        MotionGestureTransformerConfig(
            input_dim=dataset.landmark_dim,
            global_feature_dim=dataset.global_feature_dim,
            num_labels=dataset.num_labels,
            num_phases=dataset.num_phases,
            spatial_d_model=32,
            spatial_layers=1,
            spatial_heads=4,
            spatial_ffn_dim=64,
            temporal_d_model=32,
            temporal_layers=1,
            temporal_heads=4,
            temporal_ffn_dim=64,
            max_sequence_length=4,
            dropout=0.0,
        )
    )
    criterion = MotionGestureLoss(MotionGestureLossConfig(primary_loss="bce"))

    outputs = model(batch)
    loss, loss_parts = criterion(outputs, batch)

    assert outputs["action_logits"].shape == (2, len(MOTION_LABELS))
    assert outputs["phase_logits"].shape == (2, len(MOTION_LABELS), dataset.num_phases)
    assert outputs["direction_logits"].shape == (2, 8)
    assert outputs["displacement"].shape == (2, 3)
    assert loss.item() > 0
    assert loss_parts["total"] > 0


def test_motion_training_dry_run_writes_summary(tmp_path: Path) -> None:
    from training.jobs.train_motion import train_motion

    csv_path = write_motion_csv(tmp_path)
    output_root = tmp_path / "runs"

    summary = train_motion(
        {
            "run": {"name": "motion_test", "output_root": str(output_root), "seed": 123},
            "data": {
                "csv_path": str(csv_path),
                "sequence_length": 4,
                "window_stride": 2,
                "batch_size": 2,
                "val_fraction": 0.0,
                "test_fraction": 0.0,
            },
            "model": {
                "spatial_d_model": 32,
                "spatial_layers": 1,
                "spatial_heads": 4,
                "spatial_ffn_dim": 64,
                "temporal_d_model": 32,
                "temporal_layers": 1,
                "temporal_heads": 4,
                "temporal_ffn_dim": 64,
                "dropout": 0.0,
            },
            "training": {"max_epochs": 1, "amp": False},
            "plots": {"enabled": False},
        },
        device_name="cpu",
        dry_run=True,
    )

    run_dir = Path(summary["run_dir"])
    assert summary["dry_run"] is True
    assert summary["output_shapes"]["action_logits"] == [2, len(MOTION_LABELS)]
    assert (run_dir / "config.resolved.yaml").exists()
    assert (run_dir / "metrics" / "dataset_profile.json").exists()
    assert (run_dir / "metrics" / "dry_run_summary.json").exists()
    assert (run_dir / "phase_labels.json").exists()


def test_motion_dataloader_uses_distributed_sampler_for_train_split(tmp_path: Path) -> None:
    from torch.utils.data.distributed import DistributedSampler

    from training.datasets.motion_window_dataset import build_motion_window_dataloaders

    csv_path = write_motion_csv(tmp_path)

    dataloaders, _, _ = build_motion_window_dataloaders(
        csv_path,
        sequence_length=4,
        window_stride=2,
        batch_size=2,
        val_fraction=0.25,
        test_fraction=0.0,
        distributed_rank=1,
        distributed_world_size=2,
    )

    assert isinstance(dataloaders["train"].sampler, DistributedSampler)
    assert dataloaders["train"].sampler.rank == 1
    assert dataloaders["train"].sampler.num_replicas == 2
    assert not isinstance(dataloaders["validation"].sampler, DistributedSampler)


def test_motion_loss_grid_expands_grouped_values() -> None:
    from training.jobs.grid_search_motion_loss import expand_loss_grid

    trials = expand_loss_grid(
        [
            {"primary_loss": ["bce"], "use_pos_weight": [True, False]},
            {"primary_loss": ["focal"], "focal_alpha": [0.25, 0.5], "focal_gamma": [2.0]},
        ]
    )

    assert trials == [
        {"primary_loss": "bce", "use_pos_weight": True},
        {"primary_loss": "bce", "use_pos_weight": False},
        {"primary_loss": "focal", "focal_alpha": 0.25, "focal_gamma": 2.0},
        {"primary_loss": "focal", "focal_alpha": 0.5, "focal_gamma": 2.0},
    ]


def test_motion_loss_grid_search_runs_dry_trials(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from training.jobs.grid_search_motion_loss import run_motion_loss_grid_search

    csv_path = write_motion_csv(tmp_path)

    summary = run_motion_loss_grid_search(
        {
            "search": {"name": "motion_loss_grid_test", "output_root": str(tmp_path / "grid")},
            "trial_overrides": {
                "data": {
                    "csv_path": str(csv_path),
                    "sequence_length": 4,
                    "window_stride": 2,
                    "batch_size": 2,
                    "val_fraction": 0.0,
                    "test_fraction": 0.0,
                },
                "model": {
                    "spatial_d_model": 32,
                    "spatial_layers": 1,
                    "spatial_heads": 4,
                    "spatial_ffn_dim": 64,
                    "temporal_d_model": 32,
                    "temporal_layers": 1,
                    "temporal_heads": 4,
                    "temporal_ffn_dim": 64,
                    "dropout": 0.0,
                },
                "training": {"max_epochs": 1, "amp": False, "early_stopping_patience": 0},
                "checkpointing": {"save_best": False, "save_last": False},
                "plots": {"enabled": False},
            },
            "loss_grid": [
                {
                    "primary_loss": ["bce", "focal"],
                    "use_pos_weight": [True],
                    "focal_alpha": [0.25],
                    "focal_gamma": [2.0],
                    "phase_weight": [0.0],
                    "direction_weight": [0.0],
                    "displacement_weight": [0.0],
                    "velocity_weight": [0.0],
                    "duration_weight": [0.0],
                    "scale_weight": [0.0],
                    "rotation_weight": [0.0],
                    "draw_weight": [0.0],
                }
            ],
        },
        device_name="cpu",
        dry_run=True,
    )
    output = capsys.readouterr().out

    grid_run_dir = Path(summary["run_dir"])
    assert summary["total_trials"] == 2
    assert summary["completed_trials"] == 2
    assert summary["failed_trials"] == 0
    assert summary["score_mode"] == "min"
    assert summary["best_trial"] is not None
    assert (grid_run_dir / "grid_search_results.json").exists()
    assert (grid_run_dir / "grid_search_results.csv").exists()
    assert "loss grid search" in output
    assert "trial=1/2" in output
    assert "best trial=" in output


def write_motion_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "motion_sequence_frames.csv"
    rows = [
        *build_motion_sample_rows("sample-drag", "drag", "right", "normal", drag_active_labels),
        *build_motion_sample_rows("sample-swipe", "swipe_right", "left", "fast", swipe_active_labels),
        *build_motion_sample_rows("sample-negative", "no_gesture", "right", "slow", negative_active_labels),
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def build_motion_sample_rows(sample_id, target_name, handedness, speed, label_fn):
    elapsed_values = (0.0, 450.0, 900.0, 1400.0, 1590.0, 1800.0)
    phases = ("none", "starting", "active", "ending", "completed", "none")
    target_labels = sorted({label for phase in phases for label in label_fn(phase)})
    return [
        build_motion_row(
            sample_id,
            target_name,
            handedness,
            speed,
            frame_index,
            elapsed_ms,
            phase,
            tuple(label_fn(phase)),
            tuple(target_labels),
        )
        for frame_index, (elapsed_ms, phase) in enumerate(zip(elapsed_values, phases, strict=True))
    ]


def build_motion_row(
    sample_id: str,
    target_name: str,
    handedness: str,
    speed: str,
    frame_index: int,
    elapsed_ms: float,
    phase: str,
    active_labels: tuple[str, ...],
    target_labels: tuple[str, ...],
) -> dict[str, object]:
    row: dict[str, object] = {fieldname: "" for fieldname in FIELDNAMES}
    row.update(
        {
            "schema_version": "1",
            "sample_id": sample_id,
            "session_id": "session-1",
            "recorded_at_utc": "2026-08-24T00:00:00+00:00",
            "model_family": "motion",
            "target_name": target_name,
            "target_display": target_name.replace("_", " "),
            "target_handedness": handedness,
            "target_speed": speed,
            "target_count_for_group": 5,
            "capture_index_for_group": 1,
            "participant_id": "tester",
            "label_source": "unit-test",
            "labels": "|".join(target_labels),
            "active_labels": "|".join(active_labels),
            "phase": phase,
            "phase_progress": min(1.0, max(0.0, (elapsed_ms - 400.0) / 1200.0)),
            "is_interval_active": int(phase != "none"),
            "interval_start_time": 0.4,
            "interval_end_time": 1.6,
            "interval_start_frame": 1,
            "interval_end_frame": 4,
            "interval_duration_ms": 1200,
            "sample_duration_ms": 1800,
            "sample_frame_count": 6,
            "valid_hand_frame_count": 6,
            "missing_hand_frame_count": 0,
            "sample_frame_index": frame_index,
            "sample_elapsed_ms": elapsed_ms,
            "frame_index": frame_index,
            "frame_timestamp": elapsed_ms / 1000.0,
            "image_path": f"data/motion_sequences/images/{target_name}/{sample_id}/frame_{frame_index:04d}.jpg",
            "camera_index": 0,
            "camera_width": 640,
            "camera_height": 480,
            "camera_fps": 30,
            "mirror": 1,
            "image_width": 640,
            "image_height": 480,
            "detected_handedness": handedness,
            "detected_handedness_score": 0.95,
            "hand_selection_note": "synthetic",
            "tracking_missing": 0,
            "feature_scale": 0.1 + frame_index * 0.01,
            "feature_palm_center_x": 0.2 + frame_index * 0.03,
            "feature_palm_center_y": 0.3,
            "feature_palm_center_z": 0.0,
            "feature_pinch_distance": 0.08,
            "feature_pinch_delta": 0.01,
            "feature_velocity_x": 0.03,
            "feature_velocity_y": 0.0,
            "feature_velocity_z": 0.0,
            "feature_speed": 0.03,
            "feature_direction_x": 1.0,
            "feature_direction_y": 0.0,
            "feature_direction_z": 0.0,
            "feature_bbox_min_x": 0.1,
            "feature_bbox_min_y": 0.1,
            "feature_bbox_max_x": 0.4,
            "feature_bbox_max_y": 0.5,
            "feature_finger_curl_thumb": 0,
            "feature_finger_curl_index": 0,
            "feature_finger_curl_middle": 0,
            "feature_finger_curl_ring": 1,
            "feature_finger_curl_pinky": 1,
            "feature_extended_fingers": "index|middle",
        }
    )
    label_set = set(active_labels)
    for label in MOTION_LABELS:
        row[f"label_{label}"] = int(label in label_set)
    for landmark_index in range(21):
        for axis_index, axis in enumerate(("x", "y", "z")):
            value = frame_index * 0.01 + landmark_index * 0.001 + axis_index * 0.0001
            row[f"raw_landmark_{landmark_index:02d}_{axis}"] = value
            row[f"normalized_landmark_{landmark_index:02d}_{axis}"] = value * 0.5
    return row


def drag_active_labels(phase: str) -> tuple[str, ...]:
    return {
        "starting": ("DRAG_START",),
        "active": ("DRAG_MOVE",),
        "ending": ("DRAG_END",),
        "completed": ("DRAG_END",),
    }.get(phase, ())


def swipe_active_labels(phase: str) -> tuple[str, ...]:
    return ("SWIPE_RIGHT",) if phase != "none" else ()


def negative_active_labels(phase: str) -> tuple[str, ...]:
    return ()
