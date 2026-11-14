#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "two_handed_poses" / "two_handed_pose_frames.csv"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "data" / "two_handed_poses" / "images"
WINDOW_NAME = "HLGAM Two-Handed Pose Capture"
SCHEMA_VERSION = "1"
MODEL_FAMILY = "two_hand"
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


TWO_HAND_LABELS = (
    "TWO_HAND_ZOOM_IN",
    "TWO_HAND_ZOOM_OUT",
    "TWO_HAND_ROTATE_CLOCKWISE",
    "TWO_HAND_ROTATE_COUNTERCLOCKWISE",
    "CLAP",
    "FRAME_RECTANGLE",
    "TWO_HAND_PAN",
)

HAND_FEATURE_COLUMNS = (
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

PAIR_FEATURE_COLUMNS = (
    "pair_center_x",
    "pair_center_y",
    "pair_center_z",
    "pair_vector_x",
    "pair_vector_y",
    "pair_vector_z",
    "pair_distance",
    "pair_angle_radians",
    "pair_angle_degrees",
    "pair_bbox_min_x",
    "pair_bbox_min_y",
    "pair_bbox_max_x",
    "pair_bbox_max_y",
    "pair_bbox_width",
    "pair_bbox_height",
    "pair_scale_mean",
    "pair_scale_ratio",
    "pair_pinch_distance_mean",
    "pair_clap_distance",
)

BASE_COLUMNS = (
    "schema_version",
    "capture_id",
    "captured_at_utc",
    "model_family",
    "target_name",
    "target_display",
    "target_variant",
    "target_count_for_group",
    "capture_index_for_group",
    "participant_id",
    "label_source",
    "labels",
    "image_path",
    "camera_index",
    "camera_width",
    "camera_height",
    "camera_fps",
    "mirror",
    "frame_index",
    "frame_timestamp",
    "image_width",
    "image_height",
    "left_detected_handedness",
    "left_detected_handedness_score",
    "right_detected_handedness",
    "right_detected_handedness_score",
    "hand_selection_note",
    "tracking_missing",
    "lighting_quality",
    "background_complexity",
    "distance_from_camera",
    "notes",
    *PAIR_FEATURE_COLUMNS,
)


def landmark_columns(side: str, prefix: str) -> list[str]:
    return [f"{side}_{prefix}_landmark_{index:02d}_{axis}" for index in range(21) for axis in ("x", "y", "z")]


def hand_feature_columns(side: str) -> list[str]:
    return [f"{side}_{name}" for name in HAND_FEATURE_COLUMNS]


LABEL_COLUMNS = tuple(f"label_{label}" for label in TWO_HAND_LABELS)
FIELDNAMES = [
    *BASE_COLUMNS,
    *LABEL_COLUMNS,
    *hand_feature_columns("left"),
    *hand_feature_columns("right"),
    *landmark_columns("left", "raw"),
    *landmark_columns("left", "normalized"),
    *landmark_columns("right", "raw"),
    *landmark_columns("right", "normalized"),
]


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
class TwoHandCaptureTarget:
    name: str
    display: str
    labels: tuple[str, ...]
    instructions: str
    variant: str = ""


@dataclass(frozen=True)
class CaptureSelection:
    target: TwoHandCaptureTarget
    existing_count: int


@dataclass(frozen=True)
class HandPair:
    left: Any
    right: Any


@dataclass(frozen=True)
class PairMetrics:
    center: Any
    vector: Any
    distance: float
    angle_radians: float
    bbox: tuple[float, float, float, float]
    bbox_width: float
    bbox_height: float
    scale_mean: float
    scale_ratio: float
    pinch_distance_mean: float


@dataclass(frozen=True)
class CapturedTwoHandPose:
    frame: Any
    image_path: Path | None
    pair: HandPair
    left_normalized_hand: Any
    right_normalized_hand: Any
    left_features: Any
    right_features: Any
    hand_selection_note: str
    image_width: int
    image_height: int


@dataclass(frozen=True)
class RemovalResult:
    target_name: str | None
    capture_id: str | None
    csv_path: Path
    image_root: Path
    removed_rows: int
    removed_files: int
    dry_run: bool = False


def ensure_project_root_on_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture two-handed pose training rows as CSV records with matching image snapshots."
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Two-handed pose CSV file.")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT, help="Root folder for captured images.")
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
    parser.add_argument("--target", help="Capture only this target name or display label.")
    parser.add_argument(
        "--target-per-group",
        type=int,
        default=10,
        help="Accepted captures needed for each two-hand target group.",
    )
    parser.add_argument("--participant-id", default="anonymous", help="Participant identifier or anonymous code.")
    parser.add_argument("--label-source", default="human", help="Label source metadata, usually human.")
    parser.add_argument("--lighting-quality", default="", help="Optional lighting-quality metadata bucket.")
    parser.add_argument("--background-complexity", default="", help="Optional background-complexity metadata bucket.")
    parser.add_argument("--distance-from-camera", default="", help="Optional distance-from-camera metadata bucket.")
    parser.add_argument("--notes", default="", help="Optional notes stored on every accepted row.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for saved frame images.")
    parser.add_argument("--remove-capture", help="Remove one capture ID from the CSV and image folder, then exit.")
    parser.add_argument("--remove-target", help="Remove captures for this target name/display, then exit.")
    parser.add_argument("--dry-run", action="store_true", help="Preview a removal without deleting files or rows.")
    parser.add_argument("--yes", action="store_true", help="Do not prompt before deleting two-handed captures.")
    mirror_group = parser.add_mutually_exclusive_group()
    mirror_group.add_argument("--mirror", dest="mirror", action="store_true", default=True, help="Mirror camera input.")
    mirror_group.add_argument("--no-mirror", dest="mirror", action="store_false", help="Do not mirror camera input.")
    return parser.parse_args(argv)


def build_capture_targets() -> tuple[TwoHandCaptureTarget, ...]:
    targets = [
        TwoHandCaptureTarget(
            "neutral_pair",
            "Neutral two-hand pair",
            (),
            "Keep both hands visible in a relaxed neutral pose without performing a pair gesture.",
            "neutral",
        ),
        two_hand_target(
            "frame_rectangle",
            "Frame rectangle",
            "FRAME_RECTANGLE",
            "Use both hands in L-shapes to frame a rectangle.",
            variant="rectangle",
        ),
        two_hand_target(
            "clap_contact",
            "Clap contact",
            "CLAP",
            "Bring both palms together at the point of clap contact.",
            variant="contact",
        ),
        two_hand_target(
            "zoom_in_wide",
            "Two-hand zoom in wide pose",
            "TWO_HAND_ZOOM_IN",
            "Hold both hands far apart as the wide end-state for a zoom-in gesture.",
            variant="wide",
        ),
        two_hand_target(
            "zoom_out_close",
            "Two-hand zoom out close pose",
            "TWO_HAND_ZOOM_OUT",
            "Hold both hands close together as the close end-state for a zoom-out gesture.",
            variant="close",
        ),
        two_hand_target(
            "rotate_clockwise_diagonal",
            "Two-hand rotate clockwise diagonal",
            "TWO_HAND_ROTATE_CLOCKWISE",
            "Hold the pair on a clear clockwise diagonal orientation.",
            variant="clockwise_diagonal",
        ),
        two_hand_target(
            "rotate_counterclockwise_diagonal",
            "Two-hand rotate counterclockwise diagonal",
            "TWO_HAND_ROTATE_COUNTERCLOCKWISE",
            "Hold the pair on a clear counterclockwise diagonal orientation.",
            variant="counterclockwise_diagonal",
        ),
    ]
    for direction, display_direction in cardinal_directions().items():
        targets.append(
            two_hand_target(
                f"pan_{direction}",
                f"Two-hand pan {display_direction}",
                "TWO_HAND_PAN",
                f"Hold both hands shifted toward the {display_direction} side of the frame.",
                variant=direction,
            )
        )
    return tuple(sorted(targets, key=lambda target: target.display.casefold()))


def two_hand_target(
    name: str,
    display: str,
    label: str,
    instructions: str,
    *,
    variant: str = "",
) -> TwoHandCaptureTarget:
    if label not in TWO_HAND_LABELS:
        raise RuntimeError(f"unknown two-hand label in capture target: {label}")
    return TwoHandCaptureTarget(name, display, (label,), instructions, variant)


def cardinal_directions() -> dict[str, str]:
    return {"left": "left", "right": "right", "up": "up", "down": "down"}


CAPTURE_TARGETS = build_capture_targets()


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


def load_existing_counts(csv_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return counts
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            target_name = (row.get("target_name") or "").strip()
            if target_name:
                counts[target_name] = counts.get(target_name, 0) + 1
    return counts


def prompt_for_capture_selection(
    counts: Mapping[str, int],
    target_per_group: int,
    args: argparse.Namespace,
) -> CaptureSelection | None:
    fixed_target = resolve_capture_target(args.target) if args.target else None
    targets = (fixed_target,) if fixed_target is not None else CAPTURE_TARGETS
    available: list[CaptureSelection] = []
    for target in targets:
        existing_count = counts.get(target.name, 0)
        if existing_count < target_per_group:
            available.append(CaptureSelection(target, existing_count))

    if not available:
        print("\nAll selected two-handed pose capture groups are complete.")
        return None
    if fixed_target or not sys.stdin.isatty():
        return available[0]

    print("\nTwo-handed pose capture groups with remaining samples:")
    for selection_index, selection in enumerate(available, start=1):
        remaining = target_per_group - selection.existing_count
        labels = ", ".join(selection.target.labels) if selection.target.labels else "negative/no pair gesture"
        print(
            f"  {selection_index:2d}. {selection.target.display} "
            f"({selection.existing_count}/{target_per_group}, {remaining} remaining) -> {labels}"
        )
    print("Enter a group number, press Enter to refresh, or type q to quit.")

    while True:
        selected = input("Capture group: ").strip().lower()
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
                f"existing CSV header in {csv_path} does not match two-hand schema version {SCHEMA_VERSION}; "
                "write to a new file or migrate the existing dataset"
            )

    with csv_path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES, extrasaction="raise")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def remove_two_handed_captures(
    *,
    csv_path: Path = DEFAULT_OUTPUT_CSV,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    capture_id: str | None = None,
    target_name: str | None = None,
    dry_run: bool = False,
) -> RemovalResult:
    if not capture_id and not target_name:
        raise RuntimeError("provide --remove-capture or --remove-target")
    target = resolve_capture_target(target_name) if target_name else None
    kept_rows: list[dict[str, str]] = []
    removed_rows: list[dict[str, str]] = []
    header = FIELDNAMES

    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            header = reader.fieldnames or FIELDNAMES
            for row in reader:
                if row_matches_removal(row, capture_id, target):
                    removed_rows.append(row)
                else:
                    kept_rows.append(row)

    target_dir = image_root / target.name if target else None
    files_to_remove = collect_capture_files(removed_rows, image_root, target_dir)
    result = RemovalResult(
        target_name=target.name if target else None,
        capture_id=capture_id,
        csv_path=csv_path,
        image_root=image_root,
        removed_rows=len(removed_rows),
        removed_files=len(files_to_remove),
        dry_run=dry_run,
    )
    if dry_run:
        return result

    for image_path in sorted(files_to_remove):
        image_path.unlink(missing_ok=True)
    remove_empty_image_dirs(image_root, files_to_remove)
    if csv_path.exists():
        rewrite_csv(csv_path, header, kept_rows)
    return result


def row_matches_removal(
    row: Mapping[str, str],
    capture_id: str | None,
    target: TwoHandCaptureTarget | None,
) -> bool:
    if capture_id and (row.get("capture_id") or "") != capture_id:
        return False
    if target and (row.get("target_name") or "") != target.name:
        return False
    return True


def collect_capture_files(
    rows: list[dict[str, str]],
    image_root: Path,
    target_dir: Path | None = None,
) -> set[Path]:
    files: set[Path] = set()
    image_root_resolved = image_root.resolve()
    if target_dir is not None and target_dir.exists():
        files.update(path.resolve() for path in target_dir.rglob("*") if path.is_file())
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


def remove_empty_image_dirs(image_root: Path, removed_files: set[Path]) -> None:
    candidate_dirs = sorted({path.parent for path in removed_files}, key=lambda path: len(path.parts), reverse=True)
    for directory in candidate_dirs:
        if directory == image_root or not is_relative_to(directory, image_root):
            continue
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()


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
    preview = remove_two_handed_captures(
        csv_path=args.output_csv,
        image_root=args.image_root,
        capture_id=args.remove_capture,
        target_name=args.remove_target,
        dry_run=True,
    )
    action = "Would remove" if args.dry_run else "Preparing to remove"
    print(f"{action} {preview.removed_rows} CSV rows and {preview.removed_files} image files.")
    print(f"CSV: {preview.csv_path}")
    print(f"Image root: {preview.image_root}")
    if args.dry_run:
        return 0
    if not args.yes:
        confirmation = input("Type DELETE to permanently remove these two-handed captures: ").strip()
        if confirmation != "DELETE":
            print("Removal cancelled.")
            return 0
    result = remove_two_handed_captures(
        csv_path=args.output_csv,
        image_root=args.image_root,
        capture_id=args.remove_capture,
        target_name=args.remove_target,
    )
    print(f"Removed {result.removed_rows} CSV rows and {result.removed_files} image files.")
    return 0


def resolve_capture_target(group_name: str) -> TwoHandCaptureTarget:
    normalized = normalize_group_name(group_name)
    for target in CAPTURE_TARGETS:
        if normalized in {normalize_group_name(target.name), normalize_group_name(target.display)}:
            return target
    known_groups = ", ".join(target.name for target in CAPTURE_TARGETS)
    raise RuntimeError(f"unknown two-handed pose target '{group_name}'. Known groups: {known_groups}")


def normalize_group_name(group_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", group_name.casefold()).strip("_")


def selected_hand_pair(hands: list[Any]) -> tuple[HandPair | None, str]:
    if len(hands) < 2:
        return None, "need_two_hands_detected"

    left_candidates = [hand for hand in hands if (hand.handedness or "").casefold() == "left"]
    right_candidates = [hand for hand in hands if (hand.handedness or "").casefold() == "right"]
    if left_candidates and right_candidates:
        left_hand = max(left_candidates, key=lambda hand: hand.score or 0.0)
        right_hand = max(right_candidates, key=lambda hand: hand.score or 0.0)
        return HandPair(left_hand, right_hand), "matched_detected_handedness_pair"

    selected = sorted(hands, key=lambda hand: hand.score or 0.0, reverse=True)[:2]
    ordered = sorted(selected, key=screen_x_for_hand)
    return HandPair(ordered[0], ordered[1]), "selected_top_two_by_score_ordered_by_screen_x"


def screen_x_for_hand(hand: Any) -> float:
    return sum(point.x for point in hand.landmarks) / len(hand.landmarks)


def relative_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_captured_pose(
    frame: Any,
    image_path: Path | None,
    pair: HandPair,
    hand_selection_note: str,
) -> CapturedTwoHandPose:
    ensure_project_root_on_path()
    from hlgam.processing.feature_extractor import FeatureExtractor
    from hlgam.processing.normalization import HandNormalizer

    normalizer = HandNormalizer()
    left_normalized = normalizer.normalize(pair.left)
    right_normalized = normalizer.normalize(pair.right)
    left_features = FeatureExtractor().extract(left_normalized, raw_landmarks=pair.left)
    right_features = FeatureExtractor().extract(right_normalized, raw_landmarks=pair.right)
    image_height, image_width = frame.image.shape[:2]
    return CapturedTwoHandPose(
        frame=frame,
        image_path=image_path,
        pair=pair,
        left_normalized_hand=left_normalized,
        right_normalized_hand=right_normalized,
        left_features=left_features,
        right_features=right_features,
        hand_selection_note=hand_selection_note,
        image_width=image_width,
        image_height=image_height,
    )


def two_handed_pose_row(
    captured_pose: CapturedTwoHandPose,
    selection: CaptureSelection,
    args: argparse.Namespace,
    *,
    camera_index: int,
    capture_id: str,
    captured_at_utc: str,
) -> dict[str, object]:
    metrics = pair_metrics(captured_pose.left_features, captured_pose.right_features)
    row: dict[str, object] = {column: "" for column in FIELDNAMES}
    row.update(
        {
            "schema_version": SCHEMA_VERSION,
            "capture_id": capture_id,
            "captured_at_utc": captured_at_utc,
            "model_family": MODEL_FAMILY,
            "target_name": selection.target.name,
            "target_display": selection.target.display,
            "target_variant": selection.target.variant,
            "target_count_for_group": args.target_per_group,
            "capture_index_for_group": selection.existing_count + 1,
            "participant_id": args.participant_id,
            "label_source": args.label_source,
            "labels": "|".join(selection.target.labels),
            "image_path": relative_path(captured_pose.image_path),
            "camera_index": camera_index,
            "camera_width": args.width,
            "camera_height": args.height,
            "camera_fps": args.fps,
            "mirror": int(bool(args.mirror)),
            "frame_index": captured_pose.frame.index,
            "frame_timestamp": captured_pose.frame.timestamp,
            "image_width": captured_pose.image_width,
            "image_height": captured_pose.image_height,
            "left_detected_handedness": captured_pose.pair.left.handedness or "",
            "left_detected_handedness_score": captured_pose.pair.left.score or "",
            "right_detected_handedness": captured_pose.pair.right.handedness or "",
            "right_detected_handedness_score": captured_pose.pair.right.score or "",
            "hand_selection_note": captured_pose.hand_selection_note,
            "tracking_missing": 0,
            "lighting_quality": args.lighting_quality,
            "background_complexity": args.background_complexity,
            "distance_from_camera": args.distance_from_camera,
            "notes": args.notes,
        }
    )
    add_pair_feature_values(row, metrics)
    add_hand_feature_values(row, "left", captured_pose.left_features)
    add_hand_feature_values(row, "right", captured_pose.right_features)
    for label in TWO_HAND_LABELS:
        row[f"label_{label}"] = int(label in selection.target.labels)
    add_landmark_values(row, "left", "raw", captured_pose.pair.left.landmarks)
    add_landmark_values(row, "left", "normalized", captured_pose.left_normalized_hand.landmarks)
    add_landmark_values(row, "right", "raw", captured_pose.pair.right.landmarks)
    add_landmark_values(row, "right", "normalized", captured_pose.right_normalized_hand.landmarks)
    return row


def pair_metrics(left_features: Any, right_features: Any) -> PairMetrics:
    ensure_project_root_on_path()
    from hlgam.models import Point3D

    left_center = left_features.palm_center
    right_center = right_features.palm_center
    vector = right_center.subtract(left_center)
    center = Point3D(
        (left_center.x + right_center.x) / 2.0,
        (left_center.y + right_center.y) / 2.0,
        (left_center.z + right_center.z) / 2.0,
    )
    all_points = [*left_features.raw_landmarks.landmarks, *right_features.raw_landmarks.landmarks]
    xs = [point.x for point in all_points]
    ys = [point.y for point in all_points]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    min_scale = max(min(left_features.scale, right_features.scale), 1e-6)
    max_scale = max(left_features.scale, right_features.scale)
    return PairMetrics(
        center=center,
        vector=vector,
        distance=left_center.distance_to(right_center),
        angle_radians=math.atan2(vector.y, vector.x),
        bbox=bbox,
        bbox_width=bbox[2] - bbox[0],
        bbox_height=bbox[3] - bbox[1],
        scale_mean=(left_features.scale + right_features.scale) / 2.0,
        scale_ratio=max_scale / min_scale,
        pinch_distance_mean=(left_features.pinch_distance + right_features.pinch_distance) / 2.0,
    )


def add_pair_feature_values(row: dict[str, object], metrics: PairMetrics) -> None:
    bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = metrics.bbox
    row.update(
        {
            "pair_center_x": metrics.center.x,
            "pair_center_y": metrics.center.y,
            "pair_center_z": metrics.center.z,
            "pair_vector_x": metrics.vector.x,
            "pair_vector_y": metrics.vector.y,
            "pair_vector_z": metrics.vector.z,
            "pair_distance": metrics.distance,
            "pair_angle_radians": metrics.angle_radians,
            "pair_angle_degrees": math.degrees(metrics.angle_radians),
            "pair_bbox_min_x": bbox_min_x,
            "pair_bbox_min_y": bbox_min_y,
            "pair_bbox_max_x": bbox_max_x,
            "pair_bbox_max_y": bbox_max_y,
            "pair_bbox_width": metrics.bbox_width,
            "pair_bbox_height": metrics.bbox_height,
            "pair_scale_mean": metrics.scale_mean,
            "pair_scale_ratio": metrics.scale_ratio,
            "pair_pinch_distance_mean": metrics.pinch_distance_mean,
            "pair_clap_distance": metrics.distance,
        }
    )


def add_hand_feature_values(row: dict[str, object], side: str, features: Any) -> None:
    ensure_project_root_on_path()
    from hlgam.gestures.pose import extended_fingers

    bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = features.bounding_box
    row.update(
        {
            f"{side}_feature_scale": features.scale,
            f"{side}_feature_palm_center_x": features.palm_center.x,
            f"{side}_feature_palm_center_y": features.palm_center.y,
            f"{side}_feature_palm_center_z": features.palm_center.z,
            f"{side}_feature_pinch_distance": features.pinch_distance,
            f"{side}_feature_pinch_delta": features.pinch_delta,
            f"{side}_feature_velocity_x": features.velocity.x,
            f"{side}_feature_velocity_y": features.velocity.y,
            f"{side}_feature_velocity_z": features.velocity.z,
            f"{side}_feature_speed": features.speed,
            f"{side}_feature_direction_x": features.direction.x,
            f"{side}_feature_direction_y": features.direction.y,
            f"{side}_feature_direction_z": features.direction.z,
            f"{side}_feature_bbox_min_x": bbox_min_x,
            f"{side}_feature_bbox_min_y": bbox_min_y,
            f"{side}_feature_bbox_max_x": bbox_max_x,
            f"{side}_feature_bbox_max_y": bbox_max_y,
            f"{side}_feature_finger_curl_thumb": int(features.finger_curl.get("thumb", True)),
            f"{side}_feature_finger_curl_index": int(features.finger_curl.get("index", True)),
            f"{side}_feature_finger_curl_middle": int(features.finger_curl.get("middle", True)),
            f"{side}_feature_finger_curl_ring": int(features.finger_curl.get("ring", True)),
            f"{side}_feature_finger_curl_pinky": int(features.finger_curl.get("pinky", True)),
            f"{side}_feature_extended_fingers": "|".join(extended_fingers(features)),
        }
    )


def add_landmark_values(row: dict[str, object], side: str, prefix: str, landmarks: Sequence[Any]) -> None:
    for landmark_index, point in enumerate(landmarks):
        row[f"{side}_{prefix}_landmark_{landmark_index:02d}_x"] = point.x
        row[f"{side}_{prefix}_landmark_{landmark_index:02d}_y"] = point.y
        row[f"{side}_{prefix}_landmark_{landmark_index:02d}_z"] = point.z


def capture_row(
    frame: Any,
    image: Any,
    pair: HandPair,
    selection: CaptureSelection,
    hand_selection_note: str,
    args: argparse.Namespace,
    camera_index: int,
) -> dict[str, object]:
    import cv2

    capture_id = uuid.uuid4().hex
    image_path = args.image_root / selection.target.name / f"{capture_id}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
    captured_pose = build_captured_pose(frame, image_path, pair, hand_selection_note)
    return two_handed_pose_row(
        captured_pose,
        selection,
        args,
        camera_index=camera_index,
        capture_id=capture_id,
        captured_at_utc=datetime.now(UTC).isoformat(),
    )


def draw_hand_preview(image: Any, hand: Any, color: tuple[int, int, int], label: str) -> None:
    import cv2

    image_height, image_width = image.shape[:2]
    points = [(int(point.x * image_width), int(point.y * image_height)) for point in hand.landmarks]
    for start_index, end_index in HAND_CONNECTIONS:
        cv2.line(image, points[start_index], points[end_index], color, 2)
    for point in points:
        cv2.circle(image, point, 4, (0, 255, 0), -1)
    cv2.putText(image, label, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_detected_hands(image: Any, hands: list[Any], pair: HandPair | None) -> None:
    for hand in hands:
        label = hand.handedness or "unknown"
        color = (180, 180, 180)
        if pair is not None and hand is pair.left:
            label = "left slot"
            color = (0, 200, 255)
        elif pair is not None and hand is pair.right:
            label = "right slot"
            color = (255, 160, 0)
        draw_hand_preview(image, hand, color, label)


def draw_status_overlay(
    image: Any,
    selection: CaptureSelection,
    captured_in_group: int,
    target_per_group: int,
    pair: HandPair | None,
    hand_selection_note: str,
) -> None:
    total_count = selection.existing_count + captured_in_group
    labels = ", ".join(selection.target.labels) if selection.target.labels else "negative/no pair gesture"
    detected = "none"
    if pair is not None:
        detected = (
            f"left={pair.left.handedness or 'unknown'} score={pair.left.score or 0.0:.2f}  "
            f"right={pair.right.handedness or 'unknown'} score={pair.right.score or 0.0:.2f}"
        )
    lines = [
        f"Target: {selection.target.display} ({total_count}/{target_per_group})",
        f"Labels: {labels}",
        f"Detected pair: {detected} [{hand_selection_note}]",
        selection.target.instructions,
        "Space/Enter: capture  |  q/Esc: exit this group",
    ]
    draw_text_panel(image, lines)


def draw_text_panel(image: Any, lines: Sequence[str]) -> None:
    import cv2

    for line_index, line in enumerate(lines):
        y_position = 28 + line_index * 24
        cv2.putText(image, line, (16, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (16, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)


def capture_group(
    camera: Any,
    tracker: Any,
    selection: CaptureSelection,
    args: argparse.Namespace,
    camera_index: int,
) -> list[dict[str, object]]:
    import cv2

    rows: list[dict[str, object]] = []
    print(
        f"\nCapturing {selection.target.display}. "
        "Focus the OpenCV preview window, then press Space or Enter to capture."
    )
    print(f"Instruction: {selection.target.instructions}")
    try:
        while selection.existing_count + len(rows) < args.target_per_group:
            frame = camera.read()
            if frame is None:
                continue

            hands = tracker.track(frame)
            pair, hand_selection_note = selected_hand_pair(hands)
            preview_image = frame.image.copy()
            original_image = frame.image.copy()
            draw_detected_hands(preview_image, hands, pair)
            draw_status_overlay(
                preview_image,
                selection,
                len(rows),
                args.target_per_group,
                pair,
                hand_selection_note,
            )

            cv2.imshow(WINDOW_NAME, preview_image)
            key = cv2.waitKey(1) & 0xFF
            if key in EXIT_KEYS:
                break
            if key not in CAPTURE_KEYS:
                continue
            if pair is None:
                print("Two hands are required; capture skipped.")
                continue

            row = capture_row(
                frame,
                original_image,
                pair,
                selection,
                hand_selection_note,
                args,
                camera_index,
            )
            rows.append(row)
            print(f"Captured {row['capture_index_for_group']}/{args.target_per_group}: {row['image_path']}")
    finally:
        cv2.destroyWindow(WINDOW_NAME)
    return rows


def validate_args(args: argparse.Namespace) -> None:
    if args.target_per_group <= 0:
        raise RuntimeError("--target-per-group must be greater than zero")
    if args.max_hands < 2:
        raise RuntimeError("--max-hands must be at least 2 for two-handed pose capture")
    if not 1 <= args.jpeg_quality <= 100:
        raise RuntimeError("--jpeg-quality must be between 1 and 100")


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    if args.remove_capture or args.remove_target:
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
        max_num_hands=args.max_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        model_complexity=args.model_complexity,
    )
    counts = load_existing_counts(args.output_csv)
    print(f"Writing two-handed pose CSV rows to {args.output_csv}")
    print(f"Writing captured images under {args.image_root}")

    with OpenCVCamera(camera_settings) as camera, MediaPipeHandTracker(tracker_settings) as tracker:
        while True:
            selection = prompt_for_capture_selection(counts, args.target_per_group, args)
            if selection is None:
                break
            rows = capture_group(camera, tracker, selection, args, camera_index)
            append_rows(args.output_csv, rows)
            if rows:
                counts[selection.target.name] = counts.get(selection.target.name, 0) + len(rows)
                print(f"Stored {len(rows)} rows in {args.output_csv}")
            else:
                print("No rows captured for this group.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Two-handed pose capture interrupted.", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
