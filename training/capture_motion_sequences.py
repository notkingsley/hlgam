#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "motion_sequences" / "motion_sequence_frames.csv"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "data" / "motion_sequences" / "images"
WINDOW_NAME = "HLGAM Motion Sequence Capture"
SCHEMA_VERSION = "1"
MODEL_FAMILY = "motion"
HANDEDNESSES = ("left", "right")
CAPTURE_SPEEDS = ("slow", "normal", "fast")
PHASES = ("none", "starting", "active", "ending", "completed", "cancelled")
CAPTURE_KEYS = {10, 13, 32}
EXIT_KEYS = {27, ord("q"), ord("Q")}
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


MOTION_LABELS = (
    "PINCH_START",
    "PINCH_HOLD",
    "PINCH_END",
    "PINCH_TAP",
    "DRAG_START",
    "DRAG_MOVE",
    "DRAG_END",
    "ZOOM_IN",
    "ZOOM_OUT",
    "SWIPE_LEFT",
    "SWIPE_RIGHT",
    "SWIPE_UP",
    "SWIPE_DOWN",
    "DIAGONAL_SWIPE_UP_LEFT",
    "DIAGONAL_SWIPE_UP_RIGHT",
    "DIAGONAL_SWIPE_DOWN_LEFT",
    "DIAGONAL_SWIPE_DOWN_RIGHT",
    "FLICK_LEFT",
    "FLICK_RIGHT",
    "FLICK_UP",
    "FLICK_DOWN",
    "FLICK_UP_LEFT",
    "FLICK_UP_RIGHT",
    "FLICK_DOWN_LEFT",
    "FLICK_DOWN_RIGHT",
    "CIRCLE_CLOCKWISE",
    "CIRCLE_COUNTERCLOCKWISE",
    "ROTATE_CLOCKWISE",
    "ROTATE_COUNTERCLOCKWISE",
    "SCROLL_UP",
    "SCROLL_DOWN",
    "SCROLL_LEFT",
    "SCROLL_RIGHT",
    "WAVE",
    "PUSH",
    "PULL",
    "GRAB_AND_THROW",
    "AIR_DRAW_START",
    "AIR_DRAW_MOVE",
    "AIR_DRAW_END",
)


BASE_COLUMNS = (
    "schema_version",
    "sample_id",
    "session_id",
    "recorded_at_utc",
    "model_family",
    "target_name",
    "target_display",
    "target_handedness",
    "target_speed",
    "target_count_for_group",
    "capture_index_for_group",
    "participant_id",
    "label_source",
    "labels",
    "active_labels",
    "phase",
    "phase_progress",
    "is_interval_active",
    "interval_start_time",
    "interval_end_time",
    "interval_start_frame",
    "interval_end_frame",
    "interval_duration_ms",
    "sample_duration_ms",
    "sample_frame_count",
    "valid_hand_frame_count",
    "missing_hand_frame_count",
    "sample_frame_index",
    "sample_elapsed_ms",
    "frame_index",
    "frame_timestamp",
    "image_path",
    "camera_index",
    "camera_width",
    "camera_height",
    "camera_fps",
    "mirror",
    "image_width",
    "image_height",
    "detected_handedness",
    "detected_handedness_score",
    "hand_selection_note",
    "tracking_missing",
    "lighting_quality",
    "background_complexity",
    "distance_from_camera",
    "notes",
    "feature_scale",
    "feature_palm_center_x",
    "feature_palm_center_y",
    "feature_palm_center_z",
    "feature_pinch_distance",
    "feature_pinch_delta",
    "feature_velocity_x",
    "feature_velocity_y",
    "feature_velocity_z",
    "feature_speed",
    "feature_direction_x",
    "feature_direction_y",
    "feature_direction_z",
    "feature_bbox_min_x",
    "feature_bbox_min_y",
    "feature_bbox_max_x",
    "feature_bbox_max_y",
    "feature_finger_curl_thumb",
    "feature_finger_curl_index",
    "feature_finger_curl_middle",
    "feature_finger_curl_ring",
    "feature_finger_curl_pinky",
    "feature_extended_fingers",
)


def landmark_columns(prefix: str) -> list[str]:
    return [f"{prefix}_landmark_{index:02d}_{axis}" for index in range(21) for axis in ("x", "y", "z")]


LABEL_COLUMNS = tuple(f"label_{label}" for label in MOTION_LABELS)
FIELDNAMES = [*BASE_COLUMNS, *LABEL_COLUMNS, *landmark_columns("raw"), *landmark_columns("normalized")]


@dataclass(frozen=True)
class CameraCandidate:
    index: int
    width: int
    height: int
    fps: float
    readable: bool

    def description(self) -> str:
        status = "readable" if self.readable else "opened, frame read failed"
        resolution = f"{self.width}x{self.height}" if self.width and self.height else "unknown resolution"
        fps_label = f"{self.fps:.1f}fps" if self.fps else "unknown fps"
        return f"camera index {self.index} ({resolution}, {fps_label}, {status})"


@dataclass(frozen=True)
class PhaseLabelSet:
    phase: str
    labels: tuple[str, ...]


@dataclass(frozen=True)
class MotionCaptureTarget:
    name: str
    display: str
    labels: tuple[str, ...]
    instructions: str
    phase_label_sets: tuple[PhaseLabelSet, ...] = ()

    def labels_for_phase(self, phase: str) -> tuple[str, ...]:
        if phase in {"none", "cancelled"}:
            return ()
        for label_set in self.phase_label_sets:
            if label_set.phase == phase:
                return label_set.labels
        return self.labels


@dataclass(frozen=True)
class CaptureSelection:
    target: MotionCaptureTarget
    handedness: str
    speed: str
    existing_count: int


@dataclass(frozen=True)
class RemovalResult:
    target_name: str | None
    handedness: str | None
    speed: str | None
    sample_id: str | None
    csv_path: Path
    image_root: Path
    removed_rows: int
    removed_samples: int
    removed_files: int
    dry_run: bool = False


@dataclass(frozen=True)
class CapturedMotionFrame:
    frame: Any
    image_path: Path | None
    sample_frame_index: int
    sample_elapsed_ms: float
    selected_hand: Any | None
    normalized_hand: Any | None
    features: Any | None
    hand_selection_note: str
    image_width: int
    image_height: int


def ensure_project_root_on_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture single-hand motion transformer training sequences as frame images plus CSV rows."
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Motion sequence frame CSV file.")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help="Root folder for saved frame images.",
    )
    parser.add_argument("--camera-index", type=int, help="Use a specific camera index and skip camera selection.")
    parser.add_argument("--camera-scan-limit", type=int, default=1, help="Highest camera index to probe.")
    parser.add_argument("--no-camera-prompt", action="store_true", help="Select the default camera automatically.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera frame width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera frame height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS.")
    parser.add_argument("--max-hands", type=int, default=2, help="Maximum MediaPipe hands to track.")
    parser.add_argument("--min-detection-confidence", type=float, default=0.6, help="MediaPipe detection threshold.")
    parser.add_argument("--min-tracking-confidence", type=float, default=0.6, help="MediaPipe tracking threshold.")
    parser.add_argument("--model-complexity", type=int, choices=(0, 1), default=1, help="MediaPipe model complexity.")
    parser.add_argument("--target", help="Capture only this motion target name or display label.")
    parser.add_argument("--handedness", help="Capture only this handedness: left or right.")
    parser.add_argument("--speed", choices=CAPTURE_SPEEDS, help="Capture only this speed bucket.")
    parser.add_argument(
        "--target-per-group",
        type=int,
        # default=15,
        default=5,
        help="Accepted samples needed for each target/handedness/speed group.",
    )
    parser.add_argument("--participant-id", default="anonymous", help="Participant identifier or anonymous code.")
    parser.add_argument("--label-source", default="human", help="Label source metadata, usually human.")
    parser.add_argument("--lighting-quality", default="", help="Optional lighting-quality metadata bucket.")
    parser.add_argument("--background-complexity", default="", help="Optional background-complexity metadata bucket.")
    parser.add_argument("--distance-from-camera", default="", help="Optional distance-from-camera metadata bucket.")
    parser.add_argument("--notes", default="", help="Optional notes stored on every accepted frame row.")
    parser.add_argument("--countdown-seconds", type=float, default=2.0, help="Countdown before recording starts.")
    parser.add_argument("--pre-roll-ms", type=int, default=450, help="Rest frames before the action interval.")
    parser.add_argument("--action-ms", type=int, default=1200, help="Expected action interval duration.")
    parser.add_argument("--post-roll-ms", type=int, default=450, help="Rest frames after the action interval.")
    parser.add_argument("--jpeg-quality", type=int, default=92, help="JPEG quality for saved frame images.")
    parser.add_argument("--image-stride", type=int, default=1, help="Save one image every N recorded frames.")
    parser.add_argument("--auto-accept", action="store_true", help="Accept each recording without prompting.")
    parser.add_argument("--no-preview", action="store_true", help="Do not open the OpenCV preview window.")
    parser.add_argument("--remove-sample", help="Remove all CSV rows and images for a sample ID, then exit.")
    parser.add_argument("--remove-target", help="Remove captures for this target name/display, then exit.")
    parser.add_argument("--remove-handedness", help="Optional handedness filter for --remove-target.")
    parser.add_argument("--remove-speed", choices=CAPTURE_SPEEDS, help="Optional speed filter for --remove-target.")
    parser.add_argument("--dry-run", action="store_true", help="Preview a removal without deleting files or rows.")
    parser.add_argument("--yes", action="store_true", help="Do not prompt before deleting motion captures.")
    mirror_group = parser.add_mutually_exclusive_group()
    mirror_group.add_argument("--mirror", dest="mirror", action="store_true", default=True, help="Mirror camera input.")
    mirror_group.add_argument("--no-mirror", dest="mirror", action="store_false", help="Do not mirror camera input.")
    return parser.parse_args(argv)


def build_motion_capture_targets() -> tuple[MotionCaptureTarget, ...]:
    targets: list[MotionCaptureTarget] = [
        MotionCaptureTarget(
            "no_gesture",
            "No gesture / neutral movement",
            (),
            "Keep the selected hand visible and move naturally without performing a target gesture.",
        ),
        lifecycle_target(
            "pinch_hold",
            "Pinch hold lifecycle",
            "Pinch thumb and index, hold briefly, then release.",
            starting=("PINCH_START",),
            active=("PINCH_HOLD",),
            ending=("PINCH_END",),
            completed=("PINCH_END",),
        ),
        simple_target("pinch_tap", "Pinch tap", "Quickly pinch and release like a tap.", "PINCH_TAP"),
        lifecycle_target(
            "drag",
            "Pinch drag lifecycle",
            "Pinch, move the hand as if dragging an object, then release.",
            starting=("DRAG_START",),
            active=("DRAG_MOVE",),
            ending=("DRAG_END",),
            completed=("DRAG_END",),
        ),
        simple_target("zoom_in", "Zoom in", "Move the hand toward the camera or expand the gesture.", "ZOOM_IN"),
        simple_target(
            "zoom_out",
            "Zoom out",
            "Move the hand away from the camera or contract the gesture.",
            "ZOOM_OUT",
        ),
        simple_target("wave", "Wave", "Wave the selected hand side to side.", "WAVE"),
        simple_target("push", "Push", "Push the selected hand forward toward the camera.", "PUSH"),
        simple_target("pull", "Pull", "Pull the selected hand backward away from the camera.", "PULL"),
        simple_target(
            "grab_and_throw",
            "Grab and throw",
            "Close the hand as if grabbing, move decisively, then release like throwing.",
            "GRAB_AND_THROW",
        ),
        lifecycle_target(
            "air_draw",
            "Air draw lifecycle",
            "Use the index fingertip to draw a short stroke in the air.",
            starting=("AIR_DRAW_START",),
            active=("AIR_DRAW_MOVE",),
            ending=("AIR_DRAW_END",),
            completed=("AIR_DRAW_END",),
        ),
    ]
    for direction, display_direction in cardinal_directions().items():
        targets.append(
            simple_target(
                f"swipe_{direction}",
                f"Swipe {display_direction}",
                f"Move the selected hand in one clear {display_direction} swipe.",
                f"SWIPE_{direction.upper()}",
            )
        )
        targets.append(
            simple_target(
                f"flick_{direction}",
                f"Flick {display_direction}",
                f"Make a short, quick flick {display_direction}.",
                f"FLICK_{direction.upper()}",
            )
        )
        targets.append(
            simple_target(
                f"scroll_{direction}",
                f"Scroll {display_direction}",
                f"Move steadily as if scrolling {display_direction}.",
                f"SCROLL_{direction.upper()}",
            )
        )
    for direction, display_direction in diagonal_directions().items():
        label_suffix = direction.upper()
        targets.append(
            simple_target(
                f"diagonal_swipe_{direction}",
                f"Diagonal swipe {display_direction}",
                f"Swipe diagonally {display_direction} in one clean motion.",
                f"DIAGONAL_SWIPE_{label_suffix}",
            )
        )
        targets.append(
            simple_target(
                f"flick_{direction}",
                f"Flick {display_direction}",
                f"Make a short, quick diagonal flick {display_direction}.",
                f"FLICK_{label_suffix}",
            )
        )
    for rotation_name, display, label in (
        ("circle_clockwise", "Circle clockwise", "CIRCLE_CLOCKWISE"),
        ("circle_counterclockwise", "Circle counterclockwise", "CIRCLE_COUNTERCLOCKWISE"),
        ("rotate_clockwise", "Rotate clockwise", "ROTATE_CLOCKWISE"),
        ("rotate_counterclockwise", "Rotate counterclockwise", "ROTATE_COUNTERCLOCKWISE"),
    ):
        targets.append(simple_target(rotation_name, display, f"Perform a clear {display.casefold()} motion.", label))
    return tuple(sorted(targets, key=lambda target: target.display.casefold()))


def simple_target(name: str, display: str, instructions: str, label: str) -> MotionCaptureTarget:
    return lifecycle_target(
        name,
        display,
        instructions,
        starting=(label,),
        active=(label,),
        ending=(label,),
        completed=(label,),
    )


def lifecycle_target(
    name: str,
    display: str,
    instructions: str,
    *,
    starting: tuple[str, ...],
    active: tuple[str, ...],
    ending: tuple[str, ...],
    completed: tuple[str, ...],
) -> MotionCaptureTarget:
    labels = unique_labels((*starting, *active, *ending, *completed))
    return MotionCaptureTarget(
        name,
        display,
        labels,
        instructions,
        phase_label_sets=(
            PhaseLabelSet("starting", starting),
            PhaseLabelSet("active", active),
            PhaseLabelSet("ending", ending),
            PhaseLabelSet("completed", completed),
        ),
    )


def unique_labels(labels: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for label in labels:
        if label not in MOTION_LABELS:
            raise RuntimeError(f"unknown motion label in capture target: {label}")
        if label not in result:
            result.append(label)
    return tuple(result)


def cardinal_directions() -> dict[str, str]:
    return {"left": "left", "right": "right", "up": "up", "down": "down"}


def diagonal_directions() -> dict[str, str]:
    return {
        "up_left": "up-left",
        "up_right": "up-right",
        "down_left": "down-left",
        "down_right": "down-right",
    }


CAPTURE_TARGETS = build_motion_capture_targets()


def detect_present_cameras(scan_limit: int) -> list[CameraCandidate]:
    import cv2

    candidates: list[CameraCandidate] = []
    for camera_index in range(max(0, scan_limit) + 1):
        capture = cv2.VideoCapture(camera_index)
        try:
            if not capture.isOpened():
                continue
            readable, frame = capture.read()
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if readable and frame is not None:
                height, width = frame.shape[:2]
            candidates.append(CameraCandidate(camera_index, width, height, fps, readable))
        finally:
            capture.release()
    return candidates


def prompt_for_camera(candidates: list[CameraCandidate], default_index: int) -> int:
    candidate_indices = {candidate.index for candidate in candidates}
    print("\nDetected cameras:")
    for candidate in candidates:
        marker = "*" if candidate.index == default_index else " "
        print(f"  {marker} {candidate.description()}")
    print("Enter a camera index from the list, press Enter for the default, or type q to quit.")

    while True:
        selected = input(f"Select camera index [{default_index}]: ").strip().lower()
        if selected == "":
            return default_index
        if selected in {"q", "quit", "exit"}:
            raise RuntimeError("camera selection cancelled")
        try:
            camera_index = int(selected)
        except ValueError:
            print("Please enter a numeric camera index, or q to quit.")
            continue
        if camera_index in candidate_indices:
            return camera_index
        print("That camera index was not detected. Choose one from the list.")


def resolve_camera_index(args: argparse.Namespace) -> int:
    if args.camera_index is not None:
        return args.camera_index

    candidates = detect_present_cameras(args.camera_scan_limit)
    if not candidates:
        print("No cameras detected; falling back to camera index 0.", file=sys.stderr)
        return 0

    default_candidate = next((candidate for candidate in candidates if candidate.readable), candidates[0])
    if args.no_camera_prompt or not sys.stdin.isatty():
        print(f"Selected {default_candidate.description()}")
        return default_candidate.index
    return prompt_for_camera(candidates, default_candidate.index)


def load_existing_sample_counts(csv_path: Path) -> dict[tuple[str, str, str], int]:
    counts: dict[tuple[str, str, str], set[str]] = {}
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            target_name = (row.get("target_name") or "").strip()
            target_handedness = (row.get("target_handedness") or "").strip().casefold()
            target_speed = (row.get("target_speed") or "").strip().casefold()
            sample_id = (row.get("sample_id") or "").strip()
            if target_name and target_handedness and target_speed and sample_id:
                counts.setdefault((target_name, target_handedness, target_speed), set()).add(sample_id)
    return {key: len(sample_ids) for key, sample_ids in counts.items()}


def prompt_for_capture_selection(
    counts: Mapping[tuple[str, str, str], int],
    target_per_group: int,
    args: argparse.Namespace,
) -> CaptureSelection | None:
    fixed_target = resolve_capture_target(args.target) if args.target else None
    fixed_handedness = normalize_handedness(args.handedness) if args.handedness else None
    fixed_speed = args.speed
    available: list[CaptureSelection] = []
    targets = (fixed_target,) if fixed_target is not None else CAPTURE_TARGETS
    handednesses = (fixed_handedness,) if fixed_handedness is not None else HANDEDNESSES
    speeds = (fixed_speed,) if fixed_speed is not None else CAPTURE_SPEEDS
    for target in targets:
        for handedness in handednesses:
            for speed in speeds:
                existing_count = counts.get((target.name, handedness, speed), 0)
                if existing_count < target_per_group:
                    available.append(CaptureSelection(target, handedness, speed, existing_count))

    if not available:
        print("\nAll selected motion capture groups are complete.")
        return None
    if fixed_target and fixed_handedness and fixed_speed:
        return available[0]
    if not sys.stdin.isatty():
        return available[0]

    print("\nMotion capture groups with remaining samples:")
    for selection_index, selection in enumerate(available, start=1):
        remaining = target_per_group - selection.existing_count
        labels = ", ".join(selection.target.labels) if selection.target.labels else "negative/no gesture"
        print(
            f"  {selection_index:2d}. {selection.target.display} / {selection.handedness} / {selection.speed} "
            f"({selection.existing_count}/{target_per_group}, {remaining} remaining) -> {labels}"
        )
    print("Enter a group number to capture all remaining samples in that group, press Enter to refresh, or type q.")

    while True:
        selected = input("Capture group to fill: ").strip().lower()
        if selected == "":
            return prompt_for_capture_selection(counts, target_per_group, args)
        if selected in {"q", "quit", "exit"}:
            return None
        try:
            selection_index = int(selected)
        except ValueError:
            print("Please enter a group number, Enter, or q.")
            continue
        if 1 <= selection_index <= len(available):
            return available[selection_index - 1]
        print("That group number is not available.")


def capture_group_key(selection: CaptureSelection) -> tuple[str, str, str]:
    return (selection.target.name, selection.handedness, selection.speed)


def append_rows(csv_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.reader(csv_file)
            existing_header = next(reader, [])
        if existing_header != FIELDNAMES:
            raise RuntimeError(
                f"existing CSV header in {csv_path} does not match motion schema version {SCHEMA_VERSION}; "
                "write to a new file or migrate the existing dataset"
            )

    with csv_path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES, extrasaction="raise")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def remove_motion_captures(
    *,
    csv_path: Path = DEFAULT_OUTPUT_CSV,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    sample_id: str | None = None,
    target_name: str | None = None,
    handedness: str | None = None,
    speed: str | None = None,
    dry_run: bool = False,
) -> RemovalResult:
    if not sample_id and not target_name:
        raise RuntimeError("provide --remove-sample or --remove-target")
    target = resolve_capture_target(target_name) if target_name else None
    normalized_handedness = normalize_handedness(handedness) if handedness else None
    kept_rows: list[dict[str, str]] = []
    removed_rows: list[dict[str, str]] = []
    header = FIELDNAMES

    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            header = reader.fieldnames or FIELDNAMES
            for row in reader:
                if row_matches_removal(row, sample_id, target, normalized_handedness, speed):
                    removed_rows.append(row)
                else:
                    kept_rows.append(row)

    sample_ids = sorted({row.get("sample_id", "") for row in removed_rows if row.get("sample_id")})
    files_to_remove = collect_sample_files(removed_rows, image_root)
    result = RemovalResult(
        target_name=target.name if target else None,
        handedness=normalized_handedness,
        speed=speed,
        sample_id=sample_id,
        csv_path=csv_path,
        image_root=image_root,
        removed_rows=len(removed_rows),
        removed_samples=len(sample_ids),
        removed_files=len(files_to_remove),
        dry_run=dry_run,
    )
    if dry_run:
        return result

    for image_path in sorted(files_to_remove):
        image_path.unlink(missing_ok=True)
    remove_empty_sample_dirs(image_root, sample_ids)
    if csv_path.exists():
        rewrite_csv(csv_path, header, kept_rows)
    return result


def row_matches_removal(
    row: Mapping[str, str],
    sample_id: str | None,
    target: MotionCaptureTarget | None,
    handedness: str | None,
    speed: str | None,
) -> bool:
    if sample_id and (row.get("sample_id") or "") != sample_id:
        return False
    if target and (row.get("target_name") or "") != target.name:
        return False
    if handedness and (row.get("target_handedness") or "").casefold() != handedness:
        return False
    if speed and (row.get("target_speed") or "").casefold() != speed:
        return False
    return True


def collect_sample_files(rows: list[dict[str, str]], image_root: Path) -> set[Path]:
    files: set[Path] = set()
    image_root_resolved = image_root.resolve()
    for row in rows:
        image_path_value = (row.get("image_path") or "").strip()
        if not image_path_value:
            continue
        image_path = Path(image_path_value)
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path
        image_path = image_path.resolve()
        if image_path.is_file() and is_relative_to(image_path, image_root_resolved):
            files.add(image_path)
    return files


def remove_empty_sample_dirs(image_root: Path, sample_ids: Sequence[str]) -> None:
    for sample_id in sample_ids:
        for sample_dir in image_root.rglob(sample_id):
            if sample_dir.is_dir() and not any(sample_dir.iterdir()):
                sample_dir.rmdir()


def rewrite_csv(csv_path: Path, header: Sequence[str], rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = csv_path.with_name(f"{csv_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(header), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(csv_path)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def run_removal(args: argparse.Namespace) -> int:
    preview = remove_motion_captures(
        csv_path=args.output_csv,
        image_root=args.image_root,
        sample_id=args.remove_sample,
        target_name=args.remove_target,
        handedness=args.remove_handedness,
        speed=args.remove_speed,
        dry_run=True,
    )
    action = "Would remove" if args.dry_run else "Preparing to remove"
    print(
        f"{action} {preview.removed_samples} samples, {preview.removed_rows} CSV rows, "
        f"and {preview.removed_files} image files."
    )
    print(f"CSV: {preview.csv_path}")
    print(f"Image root: {preview.image_root}")
    if args.dry_run:
        return 0
    if not args.yes:
        confirmation = input("Type DELETE to permanently remove these motion captures: ").strip()
        if confirmation != "DELETE":
            print("Removal cancelled.")
            return 0
    result = remove_motion_captures(
        csv_path=args.output_csv,
        image_root=args.image_root,
        sample_id=args.remove_sample,
        target_name=args.remove_target,
        handedness=args.remove_handedness,
        speed=args.remove_speed,
    )
    print(
        f"Removed {result.removed_samples} samples, {result.removed_rows} CSV rows, "
        f"and {result.removed_files} image files."
    )
    return 0


def resolve_capture_target(group_name: str) -> MotionCaptureTarget:
    normalized = normalize_group_name(group_name)
    for target in CAPTURE_TARGETS:
        if normalized in {normalize_group_name(target.name), normalize_group_name(target.display)}:
            return target
    known_groups = ", ".join(target.name for target in CAPTURE_TARGETS)
    raise RuntimeError(f"unknown motion capture target '{group_name}'. Known groups: {known_groups}")


def normalize_group_name(group_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", group_name.casefold()).strip("_")


def normalize_handedness(handedness: str) -> str:
    normalized = normalize_group_name(handedness)
    if normalized.endswith("_hand"):
        normalized = normalized.removesuffix("_hand")
    if normalized not in HANDEDNESSES:
        raise RuntimeError("handedness must be left or right")
    return normalized


def selected_hand_for_target(hands: list[Any], target_handedness: str) -> tuple[Any | None, str]:
    if not hands:
        return None, "no_hand_detected"
    matching_hands = [hand for hand in hands if (hand.handedness or "").casefold() == target_handedness]
    if matching_hands:
        selected = max(matching_hands, key=lambda hand: hand.score or 0.0)
        return selected, "matched_detected_handedness"
    selected = max(hands, key=lambda hand: hand.score or 0.0)
    return selected, "selected_highest_score_handedness_mismatch_or_unknown"


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def phase_for_elapsed(elapsed_ms: float, *, pre_roll_ms: int, action_ms: int, post_roll_ms: int) -> str:
    del post_roll_ms
    if elapsed_ms < pre_roll_ms:
        return "none"
    action_elapsed = elapsed_ms - pre_roll_ms
    if action_elapsed > action_ms:
        return "none"
    if action_elapsed >= action_ms:
        return "completed"
    progress = action_elapsed / max(1.0, float(action_ms))
    if progress < 0.2:
        return "starting"
    if progress < 0.8:
        return "active"
    return "ending"


def phase_progress(elapsed_ms: float, *, pre_roll_ms: int, action_ms: int) -> float:
    if elapsed_ms < pre_roll_ms:
        return 0.0
    if elapsed_ms > pre_roll_ms + action_ms:
        return 1.0
    return max(0.0, min(1.0, (elapsed_ms - pre_roll_ms) / max(1.0, float(action_ms))))


def capture_motion_sample(
    camera: Any,
    tracker: Any,
    selection: CaptureSelection,
    args: argparse.Namespace,
    camera_index: int,
    session_id: str,
) -> list[dict[str, object]]:
    import cv2

    ensure_project_root_on_path()
    from hlgam.processing.feature_extractor import FeatureExtractor
    from hlgam.processing.normalization import HandNormalizer

    sample_id = uuid.uuid4().hex
    sample_dir = args.image_root / selection.target.name / selection.handedness / selection.speed / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    total_record_ms = args.pre_roll_ms + args.action_ms + args.post_roll_ms
    print(
        f"\nCapturing {selection.target.display} / {selection.handedness} / {selection.speed}. "
        f"Sample {selection.existing_count + 1}/{args.target_per_group}."
    )
    print(f"Instruction: {selection.target.instructions}")
    print(
        f"Timing: {args.pre_roll_ms}ms rest, {args.action_ms}ms action, "
        f"{args.post_roll_ms}ms rest."
    )
    print("Focus the OpenCV preview window. Press Space/Enter to start, or q/Esc to skip this group.")

    normalizer = HandNormalizer()
    extractor = FeatureExtractor()
    if not wait_for_recording_start(camera, tracker, selection, args):
        shutil.rmtree(sample_dir, ignore_errors=True)
        return []
    run_countdown(camera, tracker, selection, args)

    frames: list[CapturedMotionFrame] = []
    start_timestamp: float | None = None
    recorded_at_utc = datetime.now(UTC).isoformat()
    while True:
        frame = camera.read()
        if frame is None:
            continue
        if start_timestamp is None:
            start_timestamp = frame.timestamp
            extractor.reset()
        sample_elapsed_ms = (frame.timestamp - start_timestamp) * 1000.0
        if sample_elapsed_ms > total_record_ms:
            break

        hands = tracker.track(frame)
        selected_hand, hand_selection_note = selected_hand_for_target(hands, selection.handedness)
        normalized_hand = None
        features = None
        if selected_hand is not None:
            normalized_hand = normalizer.normalize(selected_hand)
            features = extractor.extract(normalized_hand, raw_landmarks=selected_hand)

        sample_frame_index = len(frames)
        image_path = None
        if args.image_stride > 0 and sample_frame_index % args.image_stride == 0:
            image_path = sample_dir / f"frame_{sample_frame_index:04d}.jpg"
            cv2.imwrite(str(image_path), frame.image, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])

        image_height, image_width = frame.image.shape[:2]
        frames.append(
            CapturedMotionFrame(
                frame=frame,
                image_path=image_path,
                sample_frame_index=sample_frame_index,
                sample_elapsed_ms=sample_elapsed_ms,
                selected_hand=selected_hand,
                normalized_hand=normalized_hand,
                features=features,
                hand_selection_note=hand_selection_note,
                image_width=image_width,
                image_height=image_height,
            )
        )
        if not args.no_preview:
            preview_image = frame.image.copy()
            draw_motion_overlay(
                preview_image,
                hands,
                selected_hand,
                selection,
                args,
                sample_elapsed_ms,
                total_record_ms,
            )
            cv2.imshow(WINDOW_NAME, preview_image)
            if (cv2.waitKey(1) & 0xFF) in EXIT_KEYS:
                break

    if not frames:
        shutil.rmtree(sample_dir, ignore_errors=True)
        print("No frames recorded for this sample.")
        return []
    if not args.auto_accept and sys.stdin.isatty():
        decision = prompt_for_sample_decision(frames, selection)
        if decision == "retry":
            shutil.rmtree(sample_dir, ignore_errors=True)
            return capture_motion_sample(camera, tracker, selection, args, camera_index, session_id)
        if decision == "skip":
            shutil.rmtree(sample_dir, ignore_errors=True)
            return []
        if decision == "quit":
            shutil.rmtree(sample_dir, ignore_errors=True)
            raise KeyboardInterrupt

    rows = rows_for_sample(
        frames,
        selection,
        args,
        camera_index=camera_index,
        session_id=session_id,
        sample_id=sample_id,
        recorded_at_utc=recorded_at_utc,
    )
    write_sample_manifest(sample_dir / "sample_manifest.json", rows, selection, args)
    return rows


def wait_for_recording_start(camera: Any, tracker: Any, selection: CaptureSelection, args: argparse.Namespace) -> bool:
    import cv2

    if args.no_preview or not sys.stdin.isatty():
        return True
    while True:
        frame = camera.read()
        if frame is None:
            continue
        hands = tracker.track(frame)
        selected_hand, _ = selected_hand_for_target(hands, selection.handedness)
        preview_image = frame.image.copy()
        draw_ready_overlay(preview_image, hands, selected_hand, selection)
        cv2.imshow(WINDOW_NAME, preview_image)
        key = cv2.waitKey(1) & 0xFF
        if key in CAPTURE_KEYS:
            return True
        if key in EXIT_KEYS:
            return False


def run_countdown(camera: Any, tracker: Any, selection: CaptureSelection, args: argparse.Namespace) -> None:
    import cv2

    if args.no_preview or args.countdown_seconds <= 0:
        time.sleep(max(0.0, args.countdown_seconds))
        return
    end_time = time.perf_counter() + args.countdown_seconds
    while True:
        remaining = end_time - time.perf_counter()
        if remaining <= 0:
            return
        frame = camera.read()
        if frame is None:
            continue
        hands = tracker.track(frame)
        selected_hand, _ = selected_hand_for_target(hands, selection.handedness)
        preview_image = frame.image.copy()
        draw_countdown_overlay(preview_image, hands, selected_hand, selection, remaining)
        cv2.imshow(WINDOW_NAME, preview_image)
        if (cv2.waitKey(1) & 0xFF) in EXIT_KEYS:
            raise KeyboardInterrupt


def rows_for_sample(
    frames: Sequence[CapturedMotionFrame],
    selection: CaptureSelection,
    args: argparse.Namespace,
    *,
    camera_index: int,
    session_id: str,
    sample_id: str,
    recorded_at_utc: str,
) -> list[dict[str, object]]:
    interval_indices = [
        frame.sample_frame_index
        for frame in frames
        if phase_for_elapsed(
            frame.sample_elapsed_ms,
            pre_roll_ms=args.pre_roll_ms,
            action_ms=args.action_ms,
            post_roll_ms=args.post_roll_ms,
        ) != "none"
    ]
    interval_start_frame = min(interval_indices) if interval_indices else ""
    interval_end_frame = max(interval_indices) if interval_indices else ""
    valid_hand_frame_count = sum(1 for frame in frames if frame.selected_hand is not None)
    sample_duration_ms = max(frame.sample_elapsed_ms for frame in frames)
    rows: list[dict[str, object]] = []
    for captured_frame in frames:
        phase = phase_for_elapsed(
            captured_frame.sample_elapsed_ms,
            pre_roll_ms=args.pre_roll_ms,
            action_ms=args.action_ms,
            post_roll_ms=args.post_roll_ms,
        )
        if captured_frame.sample_frame_index == interval_end_frame:
            phase = "completed"
        active_labels = selection.target.labels_for_phase(phase)
        row = motion_row(
            captured_frame,
            selection,
            args,
            camera_index=camera_index,
            session_id=session_id,
            sample_id=sample_id,
            recorded_at_utc=recorded_at_utc,
            sample_duration_ms=sample_duration_ms,
            sample_frame_count=len(frames),
            valid_hand_frame_count=valid_hand_frame_count,
            interval_start_frame=interval_start_frame,
            interval_end_frame=interval_end_frame,
            phase=phase,
            active_labels=active_labels,
        )
        rows.append(row)
    return rows


def motion_row(
    captured_frame: CapturedMotionFrame,
    selection: CaptureSelection,
    args: argparse.Namespace,
    *,
    camera_index: int,
    session_id: str,
    sample_id: str,
    recorded_at_utc: str,
    sample_duration_ms: float,
    sample_frame_count: int,
    valid_hand_frame_count: int,
    interval_start_frame: int | str,
    interval_end_frame: int | str,
    phase: str,
    active_labels: tuple[str, ...],
) -> dict[str, object]:
    interval_start_seconds = args.pre_roll_ms / 1000.0
    interval_end_seconds = (args.pre_roll_ms + args.action_ms) / 1000.0
    row: dict[str, object] = {column: "" for column in FIELDNAMES}
    row.update(
        {
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "session_id": session_id,
            "recorded_at_utc": recorded_at_utc,
            "model_family": MODEL_FAMILY,
            "target_name": selection.target.name,
            "target_display": selection.target.display,
            "target_handedness": selection.handedness,
            "target_speed": selection.speed,
            "target_count_for_group": args.target_per_group,
            "capture_index_for_group": selection.existing_count + 1,
            "participant_id": args.participant_id,
            "label_source": args.label_source,
            "labels": "|".join(selection.target.labels),
            "active_labels": "|".join(active_labels),
            "phase": phase,
            "phase_progress": phase_progress(
                captured_frame.sample_elapsed_ms,
                pre_roll_ms=args.pre_roll_ms,
                action_ms=args.action_ms,
            ),
            "is_interval_active": int(phase != "none"),
            "interval_start_time": interval_start_seconds,
            "interval_end_time": interval_end_seconds,
            "interval_start_frame": interval_start_frame,
            "interval_end_frame": interval_end_frame,
            "interval_duration_ms": args.action_ms,
            "sample_duration_ms": sample_duration_ms,
            "sample_frame_count": sample_frame_count,
            "valid_hand_frame_count": valid_hand_frame_count,
            "missing_hand_frame_count": sample_frame_count - valid_hand_frame_count,
            "sample_frame_index": captured_frame.sample_frame_index,
            "sample_elapsed_ms": captured_frame.sample_elapsed_ms,
            "frame_index": captured_frame.frame.index,
            "frame_timestamp": captured_frame.frame.timestamp,
            "image_path": relative_path(captured_frame.image_path) if captured_frame.image_path else "",
            "camera_index": camera_index,
            "camera_width": args.width,
            "camera_height": args.height,
            "camera_fps": args.fps,
            "mirror": int(bool(args.mirror)),
            "image_width": captured_frame.image_width,
            "image_height": captured_frame.image_height,
            "hand_selection_note": captured_frame.hand_selection_note,
            "tracking_missing": int(captured_frame.selected_hand is None),
            "lighting_quality": args.lighting_quality,
            "background_complexity": args.background_complexity,
            "distance_from_camera": args.distance_from_camera,
            "notes": args.notes,
        }
    )
    if captured_frame.selected_hand is not None:
        row["detected_handedness"] = captured_frame.selected_hand.handedness or ""
        row["detected_handedness_score"] = captured_frame.selected_hand.score or ""
    if captured_frame.features is not None:
        add_feature_values(row, captured_frame.features)
    for label in MOTION_LABELS:
        row[f"label_{label}"] = int(label in active_labels)
    if captured_frame.selected_hand is not None:
        add_landmark_values(row, "raw", captured_frame.selected_hand.landmarks)
    if captured_frame.normalized_hand is not None:
        add_landmark_values(row, "normalized", captured_frame.normalized_hand.landmarks)
    return row


def add_feature_values(row: dict[str, object], features: Any) -> None:
    ensure_project_root_on_path()
    from hlgam.gestures.pose import extended_fingers

    bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = features.bounding_box
    row.update(
        {
            "feature_scale": features.scale,
            "feature_palm_center_x": features.palm_center.x,
            "feature_palm_center_y": features.palm_center.y,
            "feature_palm_center_z": features.palm_center.z,
            "feature_pinch_distance": features.pinch_distance,
            "feature_pinch_delta": features.pinch_delta,
            "feature_velocity_x": features.velocity.x,
            "feature_velocity_y": features.velocity.y,
            "feature_velocity_z": features.velocity.z,
            "feature_speed": features.speed,
            "feature_direction_x": features.direction.x,
            "feature_direction_y": features.direction.y,
            "feature_direction_z": features.direction.z,
            "feature_bbox_min_x": bbox_min_x,
            "feature_bbox_min_y": bbox_min_y,
            "feature_bbox_max_x": bbox_max_x,
            "feature_bbox_max_y": bbox_max_y,
            "feature_finger_curl_thumb": int(features.finger_curl.get("thumb", True)),
            "feature_finger_curl_index": int(features.finger_curl.get("index", True)),
            "feature_finger_curl_middle": int(features.finger_curl.get("middle", True)),
            "feature_finger_curl_ring": int(features.finger_curl.get("ring", True)),
            "feature_finger_curl_pinky": int(features.finger_curl.get("pinky", True)),
            "feature_extended_fingers": "|".join(extended_fingers(features)),
        }
    )


def add_landmark_values(row: dict[str, object], prefix: str, landmarks: Sequence[Any]) -> None:
    for landmark_index, point in enumerate(landmarks):
        row[f"{prefix}_landmark_{landmark_index:02d}_x"] = point.x
        row[f"{prefix}_landmark_{landmark_index:02d}_y"] = point.y
        row[f"{prefix}_landmark_{landmark_index:02d}_z"] = point.z


def prompt_for_sample_decision(frames: Sequence[CapturedMotionFrame], selection: CaptureSelection) -> str:
    valid_count = sum(1 for frame in frames if frame.selected_hand is not None)
    missing_count = len(frames) - valid_count
    print(
        f"Recorded {len(frames)} frames for {selection.target.display}; "
        f"tracked hand in {valid_count}, missing in {missing_count}."
    )
    print("Accept this sample? [Y]es / [r]etry / [s]kip / [q]uit")
    while True:
        decision = input("Decision [y]: ").strip().casefold()
        if decision in {"", "y", "yes", "a", "accept"}:
            return "accept"
        if decision in {"r", "retry"}:
            return "retry"
        if decision in {"s", "skip"}:
            return "skip"
        if decision in {"q", "quit", "exit"}:
            return "quit"
        print("Please enter y, r, s, or q.")


def write_sample_manifest(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    selection: CaptureSelection,
    args: argparse.Namespace,
) -> None:
    if not rows:
        return
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "sample_id": rows[0]["sample_id"],
        "session_id": rows[0]["session_id"],
        "target_name": selection.target.name,
        "target_display": selection.target.display,
        "target_handedness": selection.handedness,
        "target_speed": selection.speed,
        "labels": list(selection.target.labels),
        "frame_count": len(rows),
        "valid_hand_frame_count": rows[0]["valid_hand_frame_count"],
        "missing_hand_frame_count": rows[0]["missing_hand_frame_count"],
        "interval": {
            "start_time": rows[0]["interval_start_time"],
            "end_time": rows[0]["interval_end_time"],
            "start_frame": rows[0]["interval_start_frame"],
            "end_frame": rows[0]["interval_end_frame"],
            "duration_ms": args.action_ms,
        },
        "csv_path": relative_path(args.output_csv),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def draw_ready_overlay(image: Any, hands: list[Any], selected_hand: Any | None, selection: CaptureSelection) -> None:
    draw_detected_hands(image, hands, selected_hand)
    lines = [
        f"Ready: {selection.target.display} / {selection.handedness} / {selection.speed}",
        f"Labels: {', '.join(selection.target.labels) if selection.target.labels else 'negative/no gesture'}",
        selection.target.instructions,
        "Space/Enter: start countdown  |  q/Esc: skip",
    ]
    draw_text_panel(image, lines)


def draw_countdown_overlay(
    image: Any,
    hands: list[Any],
    selected_hand: Any | None,
    selection: CaptureSelection,
    remaining_seconds: float,
) -> None:
    draw_detected_hands(image, hands, selected_hand)
    lines = [
        f"Get ready: {selection.target.display}",
        f"Recording starts in {max(0.0, remaining_seconds):.1f}s",
        "Begin from a resting hand position.",
    ]
    draw_text_panel(image, lines)


def draw_motion_overlay(
    image: Any,
    hands: list[Any],
    selected_hand: Any | None,
    selection: CaptureSelection,
    args: argparse.Namespace,
    elapsed_ms: float,
    total_record_ms: int,
) -> None:
    draw_detected_hands(image, hands, selected_hand)
    phase = phase_for_elapsed(
        elapsed_ms,
        pre_roll_ms=args.pre_roll_ms,
        action_ms=args.action_ms,
        post_roll_ms=args.post_roll_ms,
    )
    lines = [
        f"Recording: {selection.target.display} / {selection.handedness} / {selection.speed}",
        f"Phase: {phase}  elapsed={elapsed_ms:.0f}/{total_record_ms}ms",
        selection.target.instructions if phase != "none" else "Rest / neutral hand position.",
    ]
    draw_text_panel(image, lines)
    draw_progress_bar(image, elapsed_ms / max(1.0, float(total_record_ms)))


def draw_detected_hands(image: Any, hands: list[Any], selected_hand: Any | None) -> None:
    import cv2

    image_height, image_width = image.shape[:2]
    for hand in hands:
        color = (0, 200, 255) if hand is selected_hand else (180, 180, 180)
        points = [(int(point.x * image_width), int(point.y * image_height)) for point in hand.landmarks]
        for start_index, end_index in HAND_CONNECTIONS:
            cv2.line(image, points[start_index], points[end_index], color, 2)
        for point in points:
            cv2.circle(image, point, 4, (0, 255, 0), -1)


def draw_text_panel(image: Any, lines: Sequence[str]) -> None:
    import cv2

    for line_index, line in enumerate(lines):
        y_position = 28 + line_index * 24
        cv2.putText(image, line, (16, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (16, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)


def draw_progress_bar(image: Any, progress: float) -> None:
    import cv2

    image_height, image_width = image.shape[:2]
    margin = 18
    bar_height = 16
    y_min = image_height - margin - bar_height
    y_max = image_height - margin
    x_min = margin
    x_max = image_width - margin
    clamped = max(0.0, min(1.0, progress))
    fill_x = int(x_min + (x_max - x_min) * clamped)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (20, 20, 20), -1)
    cv2.rectangle(image, (x_min, y_min), (fill_x, y_max), (0, 200, 255), -1)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)


def validate_args(args: argparse.Namespace) -> None:
    if args.target_per_group <= 0:
        raise RuntimeError("--target-per-group must be greater than zero")
    if args.pre_roll_ms < 0 or args.action_ms <= 0 or args.post_roll_ms < 0:
        raise RuntimeError("--pre-roll-ms and --post-roll-ms must be non-negative, and --action-ms must be positive")
    if args.image_stride <= 0:
        raise RuntimeError("--image-stride must be greater than zero")
    if not 1 <= args.jpeg_quality <= 100:
        raise RuntimeError("--jpeg-quality must be between 1 and 100")


def capture_remaining_group_samples(
    camera: Any,
    tracker: Any,
    selection: CaptureSelection,
    args: argparse.Namespace,
    camera_index: int,
    session_id: str,
    counts: dict[tuple[str, str, str], int],
) -> int:
    group_key = capture_group_key(selection)
    accepted_samples = 0
    while counts.get(group_key, selection.existing_count) < args.target_per_group:
        existing_count = counts.get(group_key, selection.existing_count)
        remaining = args.target_per_group - existing_count
        current_selection = CaptureSelection(selection.target, selection.handedness, selection.speed, existing_count)
        print(
            f"\nFilling {current_selection.target.display} / {current_selection.handedness} / "
            f"{current_selection.speed}: {remaining} remaining."
        )
        rows = capture_motion_sample(camera, tracker, current_selection, args, camera_index, session_id)
        append_rows(args.output_csv, rows)
        if not rows:
            print("No rows stored for this sample. Returning to group selection.")
            break
        counts[group_key] = existing_count + 1
        accepted_samples += 1
        print(f"Stored sample {rows[0]['sample_id']} with {len(rows)} frame rows in {args.output_csv}")
    if counts.get(group_key, selection.existing_count) >= args.target_per_group:
        print(
            f"Completed {selection.target.display} / {selection.handedness} / {selection.speed} "
            f"({args.target_per_group}/{args.target_per_group})."
        )
    return accepted_samples


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    if args.remove_sample or args.remove_target:
        return run_removal(args)

    ensure_project_root_on_path()
    from hlgam.capture.camera import CameraSettings, OpenCVCamera
    from hlgam.tracking.hand_tracker import HandTrackerSettings, MediaPipeHandTracker

    camera_index = resolve_camera_index(args)
    camera_settings = CameraSettings(
        index=camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        mirror=args.mirror,
    )
    tracker_settings = HandTrackerSettings(
        max_num_hands=max(1, args.max_hands),
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        model_complexity=args.model_complexity,
    )
    session_id = uuid.uuid4().hex
    counts = load_existing_sample_counts(args.output_csv)
    print(f"Writing motion CSV rows to {args.output_csv}")
    print(f"Writing frame images under {args.image_root}")
    print(f"Session: {session_id}")

    with OpenCVCamera(camera_settings) as camera, MediaPipeHandTracker(tracker_settings) as tracker:
        while True:
            selection = prompt_for_capture_selection(counts, args.target_per_group, args)
            if selection is None:
                break
            capture_remaining_group_samples(camera, tracker, selection, args, camera_index, session_id, counts)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Motion capture interrupted.", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
