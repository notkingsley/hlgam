from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import training.capture_motion_sequences as motion_capture
from training.capture_motion_sequences import (
    CAPTURE_TARGETS,
    FIELDNAMES,
    MOTION_LABELS,
    CaptureSelection,
    CapturedMotionFrame,
    append_rows,
    capture_remaining_group_samples,
    phase_for_elapsed,
    remove_motion_captures,
    resolve_capture_target,
    rows_for_sample,
)


def test_motion_capture_targets_cover_motion_label_schema() -> None:
    covered_labels = {label for target in CAPTURE_TARGETS for label in target.labels}
    target_names = {target.name for target in CAPTURE_TARGETS}

    assert set(MOTION_LABELS) == covered_labels
    assert "no_gesture" in target_names
    assert resolve_capture_target("Pinch hold lifecycle").labels_for_phase("starting") == ("PINCH_START",)
    assert resolve_capture_target("Pinch hold lifecycle").labels_for_phase("active") == ("PINCH_HOLD",)
    assert resolve_capture_target("Pinch hold lifecycle").labels_for_phase("completed") == ("PINCH_END",)


def test_motion_phase_assignment_includes_rest_action_and_completion() -> None:
    assert phase_for_elapsed(100, pre_roll_ms=400, action_ms=1200, post_roll_ms=400) == "none"
    assert phase_for_elapsed(450, pre_roll_ms=400, action_ms=1200, post_roll_ms=400) == "starting"
    assert phase_for_elapsed(900, pre_roll_ms=400, action_ms=1200, post_roll_ms=400) == "active"
    assert phase_for_elapsed(1400, pre_roll_ms=400, action_ms=1200, post_roll_ms=400) == "ending"
    assert phase_for_elapsed(1600, pre_roll_ms=400, action_ms=1200, post_roll_ms=400) == "completed"
    assert phase_for_elapsed(1800, pre_roll_ms=400, action_ms=1200, post_roll_ms=400) == "none"


def test_motion_rows_encode_interval_labels_and_append_schema(tmp_path: Path) -> None:
    csv_path = tmp_path / "motion_sequence_frames.csv"
    image_root = tmp_path / "images"
    sample_id = "sample-123"
    args = fake_args(csv_path, image_root)
    rows = rows_for_sample(
        fake_frames(image_root, sample_id),
        CaptureSelection(resolve_capture_target("drag"), "right", "normal", existing_count=3),
        args,
        camera_index=0,
        session_id="session-abc",
        sample_id=sample_id,
        recorded_at_utc="2026-01-01T00:00:00+00:00",
    )

    append_rows(csv_path, rows)

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        written_rows = list(reader)

    assert reader.fieldnames == FIELDNAMES
    assert len(written_rows) == 6
    assert written_rows[0]["sample_id"] == sample_id
    assert written_rows[0]["phase"] == "none"
    assert written_rows[1]["phase"] == "starting"
    assert written_rows[1]["label_DRAG_START"] == "1"
    assert written_rows[2]["phase"] == "active"
    assert written_rows[2]["label_DRAG_MOVE"] == "1"
    assert written_rows[3]["phase"] == "ending"
    assert written_rows[3]["label_DRAG_END"] == "1"
    assert written_rows[4]["phase"] == "completed"
    assert written_rows[4]["label_DRAG_END"] == "1"
    assert written_rows[5]["phase"] == "none"
    assert written_rows[5]["active_labels"] == ""


def test_remove_motion_captures_removes_rows_and_images(tmp_path: Path) -> None:
    csv_path = tmp_path / "motion_sequence_frames.csv"
    image_root = tmp_path / "images"
    sample_id = "sample-123"
    args = fake_args(csv_path, image_root)
    rows = rows_for_sample(
        fake_frames(image_root, sample_id),
        CaptureSelection(resolve_capture_target("drag"), "right", "normal", existing_count=0),
        args,
        camera_index=0,
        session_id="session-abc",
        sample_id=sample_id,
        recorded_at_utc="2026-01-01T00:00:00+00:00",
    )
    for row in rows:
        image_path = Path(row["image_path"])
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"fake image")
    append_rows(csv_path, rows)

    result = remove_motion_captures(csv_path=csv_path, image_root=image_root, sample_id=sample_id)

    assert result.removed_samples == 1
    assert result.removed_rows == 6
    assert result.removed_files == 6
    assert not any(image_root.rglob("*.jpg"))
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        assert list(csv.DictReader(csv_file)) == []


def test_capture_remaining_group_samples_fills_group_without_reselecting(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "motion_sequence_frames.csv"
    image_root = tmp_path / "images"
    args = fake_args(csv_path, image_root, target_per_group=5)
    selection = CaptureSelection(resolve_capture_target("drag"), "right", "normal", existing_count=3)
    counts = {("drag", "right", "normal"): 3}
    captured_existing_counts: list[int] = []
    appended_sample_ids: list[str] = []

    def fake_capture(camera, tracker, current_selection, current_args, camera_index, session_id):
        captured_existing_counts.append(current_selection.existing_count)
        return [{"sample_id": f"sample-{current_selection.existing_count}"}]

    def fake_append(csv_path, rows):
        appended_sample_ids.append(rows[0]["sample_id"])

    monkeypatch.setattr(motion_capture, "capture_motion_sample", fake_capture)
    monkeypatch.setattr(motion_capture, "append_rows", fake_append)

    accepted_count = capture_remaining_group_samples(
        camera=object(),
        tracker=object(),
        selection=selection,
        args=args,
        camera_index=0,
        session_id="session-abc",
        counts=counts,
    )

    assert accepted_count == 2
    assert captured_existing_counts == [3, 4]
    assert appended_sample_ids == ["sample-3", "sample-4"]
    assert counts[("drag", "right", "normal")] == 5


def fake_args(csv_path: Path, image_root: Path, *, target_per_group: int = 10) -> argparse.Namespace:
    return argparse.Namespace(
        output_csv=csv_path,
        image_root=image_root,
        target_per_group=target_per_group,
        pre_roll_ms=400,
        action_ms=1200,
        post_roll_ms=400,
        width=640,
        height=480,
        fps=30,
        mirror=True,
        participant_id="tester",
        label_source="unit-test",
        lighting_quality="bright",
        background_complexity="plain",
        distance_from_camera="near",
        notes="fixture",
    )


def fake_frames(image_root: Path, sample_id: str) -> list[CapturedMotionFrame]:
    elapsed_values = (0.0, 450.0, 900.0, 1400.0, 1590.0, 1800.0)
    return [
        CapturedMotionFrame(
            frame=SimpleNamespace(index=frame_index, timestamp=1.0 + frame_index / 30.0),
            image_path=image_root / "drag" / "right" / "normal" / sample_id / f"frame_{frame_index:04d}.jpg",
            sample_frame_index=frame_index,
            sample_elapsed_ms=elapsed_ms,
            selected_hand=None,
            normalized_hand=None,
            features=None,
            hand_selection_note="unit_test_missing_hand",
            image_width=640,
            image_height=480,
        )
        for frame_index, elapsed_ms in enumerate(elapsed_values)
    ]
