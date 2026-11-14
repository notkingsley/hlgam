from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from hlgam.models import HandLandmarks
from tests.helpers import synthetic_hand
from training.capture_two_handed_poses import (
    CAPTURE_TARGETS,
    FIELDNAMES,
    TWO_HAND_LABELS,
    CaptureSelection,
    HandPair,
    append_rows,
    build_captured_pose,
    remove_two_handed_captures,
    resolve_capture_target,
    selected_hand_pair,
    two_handed_pose_row,
)


def test_two_hand_capture_targets_cover_label_schema() -> None:
    schema_path = Path(__file__).resolve().parents[1] / "training" / "label_schemas" / "two_hand_labels.json"
    schema_labels = tuple(json.loads(schema_path.read_text(encoding="utf-8"))["labels"])
    covered_labels = {label for target in CAPTURE_TARGETS for label in target.labels}
    target_names = {target.name for target in CAPTURE_TARGETS}

    assert schema_labels == TWO_HAND_LABELS
    assert set(TWO_HAND_LABELS) == covered_labels
    assert "neutral_pair" in target_names
    assert resolve_capture_target("Frame rectangle").labels == ("FRAME_RECTANGLE",)
    assert resolve_capture_target("neutral_pair").labels == ()


def test_selected_hand_pair_prefers_detected_handedness_and_falls_back_to_screen_order() -> None:
    detected_left = with_handedness(synthetic_hand(0.0, offset_x=0.25), "left", 0.91)
    detected_right = with_handedness(synthetic_hand(0.0, offset_x=-0.25), "right", 0.92)

    pair, note = selected_hand_pair([detected_right, detected_left])

    assert pair == HandPair(detected_left, detected_right)
    assert note == "matched_detected_handedness_pair"

    unknown_screen_left = with_handedness(synthetic_hand(0.0, offset_x=-0.25), None, 0.80)
    unknown_screen_right = with_handedness(synthetic_hand(0.0, offset_x=0.25), None, 0.81)

    fallback_pair, fallback_note = selected_hand_pair([unknown_screen_right, unknown_screen_left])

    assert fallback_pair == HandPair(unknown_screen_left, unknown_screen_right)
    assert fallback_note == "selected_top_two_by_score_ordered_by_screen_x"


def test_two_handed_pose_row_encodes_pair_features_and_append_schema(tmp_path: Path) -> None:
    csv_path = tmp_path / "two_handed_pose_frames.csv"
    image_root = tmp_path / "images"
    capture_id = "capture-123"
    image_path = image_root / "frame_rectangle" / f"{capture_id}.jpg"
    args = fake_args(csv_path, image_root)
    selection = CaptureSelection(resolve_capture_target("frame_rectangle"), existing_count=2)
    captured_pose = build_captured_pose(
        fake_frame(),
        image_path,
        HandPair(
            with_handedness(synthetic_hand(1.0, offset_x=-0.20), "left", 0.94),
            with_handedness(synthetic_hand(1.0, offset_x=0.20), "right", 0.95),
        ),
        "unit_test_pair",
    )
    row = two_handed_pose_row(
        captured_pose,
        selection,
        args,
        camera_index=0,
        capture_id=capture_id,
        captured_at_utc="2026-01-01T00:00:00+00:00",
    )

    append_rows(csv_path, [row])

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        written_rows = list(reader)

    assert reader.fieldnames == FIELDNAMES
    assert len(written_rows) == 1
    assert written_rows[0]["capture_id"] == capture_id
    assert written_rows[0]["model_family"] == "two_hand"
    assert written_rows[0]["target_name"] == "frame_rectangle"
    assert written_rows[0]["capture_index_for_group"] == "3"
    assert written_rows[0]["label_FRAME_RECTANGLE"] == "1"
    assert written_rows[0]["label_CLAP"] == "0"
    assert float(written_rows[0]["pair_distance"]) > 0.0
    assert written_rows[0]["left_detected_handedness"] == "left"
    assert written_rows[0]["right_detected_handedness"] == "right"
    assert written_rows[0]["left_raw_landmark_00_x"] != ""
    assert written_rows[0]["right_normalized_landmark_20_z"] != ""


def test_remove_two_handed_captures_removes_rows_and_images(tmp_path: Path) -> None:
    csv_path = tmp_path / "two_handed_pose_frames.csv"
    image_root = tmp_path / "images"
    capture_id = "capture-123"
    image_path = image_root / "frame_rectangle" / f"{capture_id}.jpg"
    args = fake_args(csv_path, image_root)
    selection = CaptureSelection(resolve_capture_target("frame_rectangle"), existing_count=0)
    captured_pose = build_captured_pose(
        fake_frame(),
        image_path,
        HandPair(
            with_handedness(synthetic_hand(1.0, offset_x=-0.20), "left", 0.94),
            with_handedness(synthetic_hand(1.0, offset_x=0.20), "right", 0.95),
        ),
        "unit_test_pair",
    )
    row = two_handed_pose_row(
        captured_pose,
        selection,
        args,
        camera_index=0,
        capture_id=capture_id,
        captured_at_utc="2026-01-01T00:00:00+00:00",
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"fake image")
    append_rows(csv_path, [row])

    result = remove_two_handed_captures(csv_path=csv_path, image_root=image_root, target_name="frame_rectangle")

    assert result.removed_rows == 1
    assert result.removed_files == 1
    assert not any(image_root.rglob("*.jpg"))
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        assert list(csv.DictReader(csv_file)) == []


def fake_args(csv_path: Path, image_root: Path, *, target_per_group: int = 10) -> argparse.Namespace:
    return argparse.Namespace(
        output_csv=csv_path,
        image_root=image_root,
        target_per_group=target_per_group,
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


def fake_frame() -> SimpleNamespace:
    return SimpleNamespace(image=SimpleNamespace(shape=(480, 640, 3)), index=7, timestamp=1.25)


def with_handedness(hand: HandLandmarks, handedness: str | None, score: float) -> HandLandmarks:
    return HandLandmarks(
        hand.landmarks,
        timestamp=hand.timestamp,
        handedness=handedness,
        score=score,
        image_size=hand.image_size,
    )
