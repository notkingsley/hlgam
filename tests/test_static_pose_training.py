from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from training.capture_static_poses import FIELDNAMES, STATIC_LABELS

pytest.importorskip("torch")
pytest.importorskip("matplotlib")


def test_static_pose_dataset_parses_capture_csv(tmp_path: Path) -> None:
    from training.datasets.static_pose_dataset import StaticPoseDataset, static_pose_collate

    csv_path = write_static_pose_csv(tmp_path)

    dataset = StaticPoseDataset(csv_path)
    sample = dataset[0]
    batch = static_pose_collate([dataset[0], dataset[1]])
    profile = dataset.profile()

    assert len(dataset) == 4
    assert sample["landmarks"].shape == (21, 3)
    assert sample["global_features"].shape == (22,)
    assert sample["labels"].shape == (len(STATIC_LABELS),)
    assert sample["labels"][dataset.label_schema.label_to_index["PEACE_SIGN"]] == 1
    assert sample["aux"]["finger_count"].item() == 2
    assert batch["landmarks"].shape == (2, 21, 3)
    assert batch["labels"].shape == (2, len(STATIC_LABELS))
    assert profile.total_rows == 4
    assert profile.label_counts["FINGER_COUNT_2"] == 2


def test_static_pose_model_and_loss_accept_dataset_batch(tmp_path: Path) -> None:
    from hlgam.ml.models.static_pose_transformer import StaticPoseTransformer, StaticPoseTransformerConfig
    from training.common.losses import StaticPoseLoss, StaticPoseLossConfig
    from training.datasets.static_pose_dataset import StaticPoseDataset, static_pose_collate

    csv_path = write_static_pose_csv(tmp_path)
    dataset = StaticPoseDataset(csv_path)
    batch = static_pose_collate([dataset[0], dataset[1]])
    model = StaticPoseTransformer(
        StaticPoseTransformerConfig(
            input_dim=dataset.landmark_dim,
            global_feature_dim=dataset.global_feature_dim,
            num_labels=dataset.num_labels,
            d_model=32,
            num_layers=1,
            num_heads=4,
            ffn_dim=64,
            dropout=0.0,
        )
    )
    criterion = StaticPoseLoss(StaticPoseLossConfig(primary_loss="bce"))

    outputs = model(batch)
    loss, loss_parts = criterion(outputs, batch)

    assert outputs["logits"].shape == (2, len(STATIC_LABELS))
    assert outputs["finger_count_logits"].shape == (2, 6)
    assert loss.item() > 0
    assert loss_parts["total"] > 0


def test_learned_static_pose_recognizer_loads_checkpoint_and_emits_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from hlgam.ml.inference.static_pose import LearnedStaticPoseRecognizer, StaticPoseModelConfig
    from hlgam.ml.models.static_pose_transformer import StaticPoseTransformer, StaticPoseTransformerConfig
    from hlgam.ml.schemas import DEFAULT_STATIC_GLOBAL_FEATURES, LandmarkFeatureSpec, LabelSchema
    from hlgam.processing.feature_extractor import FeatureExtractor
    from hlgam.processing.normalization import HandNormalizer
    from tests.helpers import synthetic_hand

    run_dir = tmp_path / "run"
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True)
    label_schema = LabelSchema.from_path(Path("training/label_schemas/static_pose_labels.json"))
    feature_spec = LandmarkFeatureSpec(global_feature_names=DEFAULT_STATIC_GLOBAL_FEATURES)
    label_schema.write(run_dir / "label_schema.json")
    (run_dir / "input_feature_schema.json").write_text(
        json.dumps(feature_spec.to_dict()),
        encoding="utf-8",
    )
    model_config = StaticPoseTransformerConfig(
        input_dim=feature_spec.per_landmark_dim,
        global_feature_dim=feature_spec.global_feature_dim,
        num_labels=len(label_schema.labels),
        d_model=32,
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
    )
    model = StaticPoseTransformer(model_config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.label_head.net[-1].bias[label_schema.label_to_index["OPEN_PALM"]] = 8.0
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"model": model_config.to_dict()},
        },
        checkpoint_path,
    )
    monkeypatch.setenv("HLGAM_STATIC_POSE_MODEL_CHECKPOINT", str(checkpoint_path))
    monkeypatch.setenv("HLGAM_STATIC_POSE_LABEL_SCHEMA", str(run_dir / "label_schema.json"))
    monkeypatch.setenv("HLGAM_STATIC_POSE_FEATURE_SCHEMA", str(run_dir / "input_feature_schema.json"))
    monkeypatch.setenv("HLGAM_STATIC_POSE_MODEL_DEVICE", "cpu")
    monkeypatch.setenv("HLGAM_STATIC_POSE_MODEL_THRESHOLD", "0.9")
    monkeypatch.setenv("HLGAM_STATIC_POSE_MODEL_MIN_HOLD_MS", "0")
    monkeypatch.setenv("HLGAM_STATIC_POSE_MODEL_COOLDOWN_MS", "0")
    raw = synthetic_hand(timestamp=1.0)
    normalizer = HandNormalizer()
    extractor = FeatureExtractor()
    features = extractor.extract(normalizer.normalize(raw), raw_landmarks=raw)
    recognizer = LearnedStaticPoseRecognizer(StaticPoseModelConfig(enabled=True))

    events = recognizer.process(features)

    assert [event.gesture for event in events] == ["OPEN_PALM"]
    assert events[0].metadata["source"] == "learned_static_pose"
    assert events[0].metadata["model_score"] > 0.9


def test_static_pose_training_dry_run_writes_summary(tmp_path: Path) -> None:
    from training.jobs.train_static_pose import train_static_pose

    csv_path = write_static_pose_csv(tmp_path)
    output_root = tmp_path / "runs"

    summary = train_static_pose(
        {
            "run": {"name": "static_pose_test", "output_root": str(output_root), "seed": 123},
            "data": {"csv_path": str(csv_path), "batch_size": 2, "val_fraction": 0.25, "test_fraction": 0.0},
            "model": {"d_model": 32, "num_layers": 1, "num_heads": 4, "ffn_dim": 64, "dropout": 0.0},
            "training": {"max_epochs": 1, "amp": False},
        },
        device_name="cpu",
        dry_run=True,
    )

    run_dir = Path(summary["run_dir"])
    assert summary["dry_run"] is True
    assert (run_dir / "config.resolved.yaml").exists()
    assert (run_dir / "metrics" / "dataset_profile.json").exists()
    assert (run_dir / "metrics" / "dry_run_summary.json").exists()


def test_static_pose_dataloader_uses_distributed_sampler_for_train_split(tmp_path: Path) -> None:
    from torch.utils.data.distributed import DistributedSampler

    from training.datasets.static_pose_dataset import build_static_pose_dataloaders

    csv_path = write_static_pose_csv(tmp_path)

    dataloaders, _, _ = build_static_pose_dataloaders(
        csv_path,
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


def test_static_pose_training_writes_completion_plots_and_prints_metrics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from training.jobs.train_static_pose import train_static_pose

    csv_path = write_static_pose_csv(tmp_path)
    output_root = tmp_path / "runs"

    summary = train_static_pose(
        {
            "run": {"name": "static_pose_plot_test", "output_root": str(output_root), "seed": 123},
            "data": {"csv_path": str(csv_path), "batch_size": 2, "val_fraction": 0.25, "test_fraction": 0.25},
            "model": {"d_model": 32, "num_layers": 1, "num_heads": 4, "ffn_dim": 64, "dropout": 0.0},
            "training": {"max_epochs": 1, "amp": False},
            "plots": {"validation_metric_names": ["macro_f1", "micro_f1"]},
        },
        device_name="cpu",
    )
    output = capsys.readouterr().out

    loss_plot = Path(summary["plots"]["loss_plot"])
    validation_metrics_plot = Path(summary["plots"]["validation_metrics_plot"])
    assert summary["test_metrics"] is not None
    assert loss_plot.exists()
    assert validation_metrics_plot.exists()
    assert loss_plot.stat().st_size > 0
    assert validation_metrics_plot.stat().st_size > 0
    assert "val_macro_precision=" in output
    assert "val_macro_recall=" in output
    assert "val_exact_match=" in output
    assert "test evaluation" in output
    assert "test_macro_precision=" in output
    assert "test_macro_recall=" in output
    assert "test_exact_match=" in output


def test_static_training_plotter_reads_jsonl_history(tmp_path: Path) -> None:
    from training.common.checkpointing import append_jsonl
    from training.common.plotting import plot_static_training_history

    run_dir = tmp_path / "run"
    for epoch in range(1, 4):
        append_jsonl(
            run_dir / "metrics" / "train_metrics.jsonl",
            {"epoch": epoch, "metrics": {"loss": 1.0 / epoch}},
        )
        append_jsonl(
            run_dir / "metrics" / "validation_metrics.jsonl",
            {
                "epoch": epoch,
                "metrics": {
                    "loss": 1.2 / epoch,
                    "macro_f1": epoch / 3,
                    "micro_f1": epoch / 4,
                },
            },
        )

    plot_paths = plot_static_training_history(run_dir, validation_metric_names=("macro_f1", "micro_f1"))

    assert plot_paths.loss_plot.exists()
    assert plot_paths.validation_metrics_plot.exists()
    assert plot_paths.loss_plot.read_bytes().startswith(b"\x89PNG")
    assert plot_paths.validation_metrics_plot.read_bytes().startswith(b"\x89PNG")


def test_static_pose_grid_search_plotter_writes_parameter_charts_and_pareto_front(tmp_path: Path) -> None:
    from training.common.checkpointing import write_json
    from training.common.plotting import plot_static_pose_grid_search_results

    results_path = tmp_path / "grid_search_results.json"
    write_json(
        results_path,
        {
            "trials": [
                grid_plot_trial("trial_001", "bce", True, 0.0, 0.72, 0.40, 0.08),
                grid_plot_trial("trial_002", "bce", False, 0.5, 0.82, 0.47, 0.09),
                grid_plot_trial("trial_003", "focal", True, 0.0, 0.78, 0.32, 0.05),
                grid_plot_trial("trial_004", "focal", False, 0.5, 0.68, 0.28, 0.04),
            ],
        },
    )

    plot_paths = plot_static_pose_grid_search_results(
        results_path,
        metric_names=("score", "test.loss", "test.macro_f1", "test.hamming_loss"),
    )

    assert set(plot_paths.parameter_metric_plots) == {"finger_count_weight", "primary_loss", "use_pos_weight"}
    assert plot_paths.pareto_front_json.exists()
    assert plot_paths.pareto_front_csv.exists()
    for plot_path in plot_paths.parameter_metric_plots.values():
        assert plot_path.exists()
        assert plot_path.read_bytes().startswith(b"\x89PNG")
    pareto_payload = json.loads(plot_paths.pareto_front_json.read_text(encoding="utf-8"))
    assert pareto_payload["metric_directions"]["test.loss"] == "min"
    assert pareto_payload["metric_directions"]["test.macro_f1"] == "max"
    assert pareto_payload["trials"]


def test_static_pose_loss_grid_expands_grouped_values() -> None:
    from training.jobs.grid_search_static_pose_loss import expand_loss_grid

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


def test_static_pose_loss_grid_search_runs_trials(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from training.jobs.grid_search_static_pose_loss import run_static_pose_loss_grid_search

    csv_path = write_static_pose_csv(tmp_path)

    summary = run_static_pose_loss_grid_search(
        {
            "search": {"name": "loss_grid_test", "output_root": str(tmp_path / "grid")},
            "trial_overrides": {
                "data": {"csv_path": str(csv_path), "batch_size": 2, "val_fraction": 0.25, "test_fraction": 0.25},
                "model": {"d_model": 32, "num_layers": 1, "num_heads": 4, "ffn_dim": 64, "dropout": 0.0},
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
                    "finger_count_weight": [0.0],
                    "point_direction_weight": [0.0],
                    "flat_direction_weight": [0.0],
                }
            ],
        },
        device_name="cpu",
    )
    output = capsys.readouterr().out

    grid_run_dir = Path(summary["run_dir"])
    assert summary["total_trials"] == 2
    assert summary["completed_trials"] == 2
    assert summary["failed_trials"] == 0
    assert summary["best_trial"] is not None
    assert (grid_run_dir / "grid_search_results.json").exists()
    assert (grid_run_dir / "grid_search_results.csv").exists()
    assert "loss grid search" in output
    assert "trial=1/2" in output
    assert "best trial=" in output


def grid_plot_trial(
    trial_id: str,
    primary_loss: str,
    use_pos_weight: bool,
    finger_count_weight: float,
    macro_f1: float,
    loss: float,
    hamming_loss: float,
) -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "status": "completed",
        "score": macro_f1,
        "score_metric": "macro_f1",
        "score_mode": "max",
        "best_metric": macro_f1,
        "loss_config": {
            "primary_loss": primary_loss,
            "use_pos_weight": use_pos_weight,
            "finger_count_weight": finger_count_weight,
        },
        "test_metrics": {
            "loss": loss,
            "macro_f1": macro_f1,
            "hamming_loss": hamming_loss,
        },
        "summary": {"best_metric": macro_f1},
        "run_dir": f"/tmp/{trial_id}",
    }


def write_static_pose_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "static_pose_frames.csv"
    rows = [
        build_static_pose_row(
            "cap-1",
            "peace_sign",
            "right",
            ("TWO_FINGER_POINT", "PEACE_SIGN", "FINGER_COUNT_2"),
            0.0,
        ),
        build_static_pose_row("cap-2", "two_finger_point", "left", ("TWO_FINGER_POINT", "FINGER_COUNT_2"), 0.1),
        build_static_pose_row("cap-3", "thumbs_up", "right", ("THUMBS_UP", "FINGER_COUNT_1"), 0.2),
        build_static_pose_row("cap-4", "open_palm_up", "left", ("OPEN_PALM", "FINGER_COUNT_5", "FLAT_HAND_UP"), 0.3),
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def build_static_pose_row(
    capture_id: str,
    primary_pose: str,
    handedness: str,
    labels: tuple[str, ...],
    offset: float,
) -> dict[str, object]:
    row: dict[str, object] = {fieldname: "" for fieldname in FIELDNAMES}
    row.update(
        {
            "schema_version": "1",
            "capture_id": capture_id,
            "captured_at_utc": "2026-07-21T00:00:00+00:00",
            "primary_pose": primary_pose,
            "target_handedness": handedness,
            "target_count_for_group": 25,
            "capture_index_for_group": 1,
            "labels": "|".join(labels),
            "image_path": f"data/static_poses/images/{primary_pose}/{handedness}/{capture_id}.jpg",
            "camera_index": 0,
            "camera_width": 1280,
            "camera_height": 720,
            "camera_fps": 30,
            "mirror": 1,
            "frame_index": 1,
            "frame_timestamp": offset,
            "image_width": 1280,
            "image_height": 720,
            "detected_handedness": handedness,
            "detected_handedness_score": 0.95,
            "hand_selection_note": "synthetic",
            "feature_scale": 0.1 + offset,
            "feature_palm_center_x": 0.2 + offset,
            "feature_palm_center_y": 0.3 + offset,
            "feature_palm_center_z": 0.0,
            "feature_pinch_distance": 0.08,
            "feature_pinch_delta": 0.0,
            "feature_velocity_x": 0.0,
            "feature_velocity_y": 0.0,
            "feature_velocity_z": 0.0,
            "feature_speed": 0.0,
            "feature_direction_x": 0.0,
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
    label_set = set(labels)
    for label in STATIC_LABELS:
        row[f"label_{label}"] = int(label in label_set)
    for landmark_index in range(21):
        for axis_index, axis in enumerate(("x", "y", "z")):
            value = offset + landmark_index * 0.01 + axis_index * 0.001
            row[f"raw_landmark_{landmark_index:02d}_{axis}"] = value
            row[f"normalized_landmark_{landmark_index:02d}_{axis}"] = value * 0.5
    return row
