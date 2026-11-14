#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "static_poses" / "static_pose_frames.csv"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "data" / "static_poses" / "images"
WINDOW_NAME = "HLGAM Static Pose Capture"
SCHEMA_VERSION = "1"
HANDEDNESSES = ("left", "right")
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


STATIC_LABELS = (
    "OPEN_PALM",
    "FIST",
    "POINT_INDEX",
    "TWO_FINGER_POINT",
    "PEACE_SIGN",
    "OK_SIGN",
    "THUMBS_UP",
    "THUMBS_DOWN",
    "L_SHAPE",
    "ROCK_SIGN",
    "HANG_LOOSE",
    "FINGER_COUNT_1",
    "FINGER_COUNT_2",
    "FINGER_COUNT_3",
    "FINGER_COUNT_4",
    "FINGER_COUNT_5",
    "FINGER_POINT_LEFT",
    "FINGER_POINT_RIGHT",
    "FINGER_POINT_UP",
    "FINGER_POINT_DOWN",
    "FINGER_POINT_UP_LEFT",
    "FINGER_POINT_UP_RIGHT",
    "FINGER_POINT_DOWN_LEFT",
    "FINGER_POINT_DOWN_RIGHT",
    "FLAT_HAND_LEFT",
    "FLAT_HAND_RIGHT",
    "FLAT_HAND_UP",
    "FLAT_HAND_DOWN",
)


BASE_COLUMNS = (
    "schema_version",
    "capture_id",
    "captured_at_utc",
    "primary_pose",
    "target_handedness",
    "target_count_for_group",
    "capture_index_for_group",
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
    "detected_handedness",
    "detected_handedness_score",
    "hand_selection_note",
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


LABEL_COLUMNS = tuple(f"label_{label}" for label in STATIC_LABELS)
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
class CaptureTarget:
    name: str
    display: str
    labels: tuple[str, ...]
    instructions: str


@dataclass(frozen=True)
class CaptureSelection:
    target: CaptureTarget
    handedness: str
    existing_count: int


@dataclass(frozen=True)
class RemovalResult:
    group_name: str
    handedness: str
    csv_path: Path
    image_dir: Path
    removed_rows: int
    removed_files: int
    dry_run: bool = False


def ensure_project_root_on_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture static pose training rows as CSV records with matching image snapshots."
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Central static pose CSV file.")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT, help="Root folder for captured images.")
    parser.add_argument("--camera-index", type=int, help="Use a specific camera index and skip camera selection.")
    parser.add_argument("--camera-scan-limit", type=int, default=1, help="Highest camera index to probe.")
    parser.add_argument(
        "--no-camera-prompt",
        action="store_true",
        help="Select the default detected camera automatically.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Requested camera frame width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera frame height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS.")
    parser.add_argument(
        "--target-per-handedness",
        type=int,
        default=25,
        help="Captures needed for each pose/handedness pair.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for saved frame images.")
    parser.add_argument("--min-detection-confidence", type=float, default=0.6, help="MediaPipe detection threshold.")
    parser.add_argument("--min-tracking-confidence", type=float, default=0.6, help="MediaPipe tracking threshold.")
    parser.add_argument("--model-complexity", type=int, choices=(0, 1), default=1, help="MediaPipe model complexity.")
    parser.add_argument("--max-hands", type=int, default=2, help="Maximum hands to track while capturing.")
    parser.add_argument(
        "--remove-group",
        help="Remove all captures for this pose group from the CSV and image folder, then exit.",
    )
    parser.add_argument(
        "--remove-handedness",
        help="Handedness to remove with --remove-group. Accepts left, right, left hand, or right hand.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview a removal without deleting files or rows.")
    parser.add_argument("--yes", action="store_true", help="Do not prompt before deleting a capture group.")
    mirror_group = parser.add_mutually_exclusive_group()
    mirror_group.add_argument("--mirror", dest="mirror", action="store_true", default=True, help="Mirror camera input.")
    mirror_group.add_argument("--no-mirror", dest="mirror", action="store_false", help="Do not mirror camera input.")
    return parser.parse_args(argv)


def build_capture_targets() -> tuple[CaptureTarget, ...]:
    targets = [
        CaptureTarget(
            "fist",
            "Fist",
            ("FIST",),
            "Close all fingers into a fist.",
        ),
        CaptureTarget(
            "two_finger_point",
            "Two-finger point",
            ("TWO_FINGER_POINT", "FINGER_COUNT_2"),
            "Extend index and middle fingers close together; keep ring and pinky curled.",
        ),
        CaptureTarget(
            "peace_sign",
            "Peace sign",
            ("TWO_FINGER_POINT", "PEACE_SIGN", "FINGER_COUNT_2"),
            "Extend and spread index and middle fingers; keep ring and pinky curled.",
        ),
        CaptureTarget(
            "ok_sign",
            "OK sign",
            ("OK_SIGN", "FINGER_COUNT_3"),
            "Touch thumb and index together; extend middle, ring, and pinky fingers.",
        ),
        CaptureTarget(
            "thumbs_up",
            "Thumbs up",
            ("THUMBS_UP", "FINGER_COUNT_1"),
            "Extend only the thumb upward.",
        ),
        CaptureTarget(
            "thumbs_down",
            "Thumbs down",
            ("THUMBS_DOWN", "FINGER_COUNT_1"),
            "Extend only the thumb downward.",
        ),
        CaptureTarget(
            "l_shape",
            "L shape",
            ("L_SHAPE", "FINGER_COUNT_2"),
            "Extend thumb and index to form an L; keep other fingers curled.",
        ),
        CaptureTarget(
            "rock_sign",
            "Rock sign",
            ("ROCK_SIGN", "FINGER_COUNT_2"),
            "Extend index and pinky; keep middle and ring curled.",
        ),
        CaptureTarget(
            "hang_loose",
            "Hang loose",
            ("HANG_LOOSE", "FINGER_COUNT_2"),
            "Extend thumb and pinky; keep index, middle, and ring curled.",
        ),
        CaptureTarget(
            "finger_count_3",
            "Finger count three",
            ("FINGER_COUNT_3",),
            "Show any clear three-finger count that is not the OK sign.",
        ),
        CaptureTarget(
            "finger_count_4",
            "Finger count four",
            ("FINGER_COUNT_4",),
            "Show any clear four-finger count.",
        ),
    ]
    for direction, display_direction in (
        ("up", "up"),
        ("down", "down"),
        ("left", "left"),
        ("right", "right"),
    ):
        targets.append(
            CaptureTarget(
                f"open_palm_{direction}",
                f"Open palm {display_direction}",
                ("OPEN_PALM", "FINGER_COUNT_5", f"FLAT_HAND_{direction.upper()}"),
                f"Open all five fingers with the hand axis pointing {display_direction}.",
            )
        )
    for direction, display_direction in (
        ("up", "up"),
        ("down", "down"),
        ("left", "left"),
        ("right", "right"),
        ("up_left", "up-left"),
        ("up_right", "up-right"),
        ("down_left", "down-left"),
        ("down_right", "down-right"),
    ):
        targets.append(
            CaptureTarget(
                f"point_index_{direction}",
                f"Index point {display_direction}",
                ("POINT_INDEX", "FINGER_COUNT_1", f"FINGER_POINT_{direction.upper()}"),
                f"Extend only the index finger and point {display_direction}.",
            )
        )
    return tuple(sorted(targets, key=lambda target: target.display.casefold()))


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


def load_existing_counts(csv_path: Path) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return counts
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            primary_pose = (row.get("primary_pose") or "").strip()
            target_handedness = (row.get("target_handedness") or "").strip().lower()
            if primary_pose and target_handedness:
                key = (primary_pose, target_handedness)
                counts[key] = counts.get(key, 0) + 1
    return counts


def prompt_for_capture_selection(
    counts: dict[tuple[str, str], int],
    target_per_handedness: int,
) -> CaptureSelection | None:
    available: list[CaptureSelection] = []
    for target in CAPTURE_TARGETS:
        for handedness in HANDEDNESSES:
            existing_count = counts.get((target.name, handedness), 0)
            if existing_count < target_per_handedness:
                available.append(CaptureSelection(target, handedness, existing_count))

    if not available:
        print("\nAll static pose capture groups are complete.")
        return None

    print("\nStatic pose capture groups with remaining samples:")
    for selection_index, selection in enumerate(available, start=1):
        remaining = target_per_handedness - selection.existing_count
        labels = ", ".join(selection.target.labels)
        print(
            f"  {selection_index:2d}. {selection.target.display} / {selection.handedness} "
            f"({selection.existing_count}/{target_per_handedness}, {remaining} remaining) -> {labels}"
        )
    print("Enter a group number, press Enter to refresh, or type q to quit.")

    while True:
        selected = input("Capture group: ").strip().lower()
        if selected == "":
            return prompt_for_capture_selection(counts, target_per_handedness)
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
                f"existing CSV header in {csv_path} does not match schema version {SCHEMA_VERSION}; "
                "write to a new file or migrate the existing dataset"
            )

    with csv_path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES, extrasaction="raise")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def remove_capture_group(
    group_name: str,
    handedness: str,
    *,
    csv_path: Path = DEFAULT_OUTPUT_CSV,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    dry_run: bool = False,
) -> RemovalResult:
    target = resolve_capture_target(group_name)
    normalized_handedness = normalize_handedness(handedness)
    image_dir = image_root / target.name / normalized_handedness
    kept_rows: list[dict[str, str]] = []
    removed_rows: list[dict[str, str]] = []
    header = FIELDNAMES

    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            header = reader.fieldnames or FIELDNAMES
            for row in reader:
                row_group = (row.get("primary_pose") or "").strip()
                row_handedness = (row.get("target_handedness") or "").strip().casefold()
                if row_group == target.name and row_handedness == normalized_handedness:
                    removed_rows.append(row)
                else:
                    kept_rows.append(row)

    files_to_remove = collect_group_files(removed_rows, image_dir, image_root)
    result = RemovalResult(
        group_name=target.name,
        handedness=normalized_handedness,
        csv_path=csv_path,
        image_dir=image_dir,
        removed_rows=len(removed_rows),
        removed_files=len(files_to_remove),
        dry_run=dry_run,
    )
    if dry_run:
        return result

    for image_path in sorted(files_to_remove):
        if is_relative_to(image_path, image_dir):
            continue
        image_path.unlink(missing_ok=True)
    if image_dir.exists():
        shutil.rmtree(image_dir)
    if csv_path.exists():
        rewrite_csv(csv_path, header, kept_rows)
    return result


def resolve_capture_target(group_name: str) -> CaptureTarget:
    normalized = normalize_group_name(group_name)
    for target in CAPTURE_TARGETS:
        if normalized in {normalize_group_name(target.name), normalize_group_name(target.display)}:
            return target
    known_groups = ", ".join(target.name for target in CAPTURE_TARGETS)
    raise RuntimeError(f"unknown static pose group '{group_name}'. Known groups: {known_groups}")


def normalize_group_name(group_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", group_name.casefold()).strip("_")


def normalize_handedness(handedness: str) -> str:
    normalized = normalize_group_name(handedness)
    if normalized.endswith("_hand"):
        normalized = normalized.removesuffix("_hand")
    if normalized not in HANDEDNESSES:
        raise RuntimeError("handedness must be left or right")
    return normalized


def collect_group_files(rows: list[dict[str, str]], image_dir: Path, image_root: Path) -> set[Path]:
    files: set[Path] = set()
    if image_dir.exists():
        files.update(path.resolve() for path in image_dir.rglob("*") if path.is_file())
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


def run_group_removal(args: argparse.Namespace) -> int:
    if not args.remove_group or not args.remove_handedness:
        raise RuntimeError("--remove-group and --remove-handedness must be supplied together")

    preview = remove_capture_group(
        args.remove_group,
        args.remove_handedness,
        csv_path=args.output_csv,
        image_root=args.image_root,
        dry_run=True,
    )
    action = "Would remove" if args.dry_run else "Preparing to remove"
    print(
        f"{action} {preview.removed_rows} CSV rows and {preview.removed_files} files for "
        f"{preview.group_name} / {preview.handedness}."
    )
    print(f"CSV: {preview.csv_path}")
    print(f"Image folder: {preview.image_dir}")
    if args.dry_run:
        return 0
    if not args.yes:
        confirmation = input("Type DELETE to permanently remove this capture group: ").strip()
        if confirmation != "DELETE":
            print("Removal cancelled.")
            return 0

    result = remove_capture_group(
        args.remove_group,
        args.remove_handedness,
        csv_path=args.output_csv,
        image_root=args.image_root,
    )
    print(
        f"Removed {result.removed_rows} CSV rows and {result.removed_files} files for "
        f"{result.group_name} / {result.handedness}."
    )
    return 0


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


def capture_row(
    frame: Any,
    image: Any,
    hand: Any,
    selection: CaptureSelection,
    capture_index_for_group: int,
    hand_selection_note: str,
    args: argparse.Namespace,
    camera_index: int,
) -> dict[str, object]:
    import cv2

    ensure_project_root_on_path()
    from hlgam.processing.feature_extractor import FeatureExtractor
    from hlgam.processing.normalization import HandNormalizer
    from hlgam.gestures.pose import extended_fingers

    capture_id = uuid.uuid4().hex
    image_path = args.image_root / selection.target.name / selection.handedness / f"{capture_id}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])

    normalizer = HandNormalizer()
    extractor = FeatureExtractor()
    normalized_hand = normalizer.normalize(hand)
    features = extractor.extract(normalized_hand, raw_landmarks=hand)
    image_height, image_width = image.shape[:2]
    bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = features.bounding_box
    label_set = set(selection.target.labels)
    row: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "capture_id": capture_id,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "primary_pose": selection.target.name,
        "target_handedness": selection.handedness,
        "target_count_for_group": args.target_per_handedness,
        "capture_index_for_group": capture_index_for_group,
        "labels": "|".join(selection.target.labels),
        "image_path": relative_path(image_path),
        "camera_index": camera_index,
        "camera_width": args.width,
        "camera_height": args.height,
        "camera_fps": args.fps,
        "mirror": int(bool(args.mirror)),
        "frame_index": frame.index,
        "frame_timestamp": frame.timestamp,
        "image_width": image_width,
        "image_height": image_height,
        "detected_handedness": hand.handedness or "",
        "detected_handedness_score": hand.score if hand.score is not None else "",
        "hand_selection_note": hand_selection_note,
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
    for label in STATIC_LABELS:
        row[f"label_{label}"] = int(label in label_set)
    add_landmark_values(row, "raw", hand.landmarks)
    add_landmark_values(row, "normalized", normalized_hand.landmarks)
    return row


def add_landmark_values(row: dict[str, object], prefix: str, landmarks: Sequence[Any]) -> None:
    for landmark_index, point in enumerate(landmarks):
        row[f"{prefix}_landmark_{landmark_index:02d}_x"] = point.x
        row[f"{prefix}_landmark_{landmark_index:02d}_y"] = point.y
        row[f"{prefix}_landmark_{landmark_index:02d}_z"] = point.z


def draw_hand_preview(image: Any, hand: Any, color: tuple[int, int, int]) -> None:
    import cv2

    image_height, image_width = image.shape[:2]
    points = [(int(point.x * image_width), int(point.y * image_height)) for point in hand.landmarks]
    for start_index, end_index in HAND_CONNECTIONS:
        cv2.line(image, points[start_index], points[end_index], color, 2)
    for point in points:
        cv2.circle(image, point, 4, (0, 255, 0), -1)


def draw_status_overlay(
    image: Any,
    selection: CaptureSelection,
    captured_in_group: int,
    target_per_handedness: int,
    selected_hand: Any | None,
    hand_selection_note: str,
) -> None:
    import cv2

    total_count = selection.existing_count + captured_in_group
    labels = ", ".join(selection.target.labels)
    detected = "none"
    if selected_hand is not None:
        detected = f"{selected_hand.handedness or 'unknown'} score={selected_hand.score or 0.0:.2f}"
    lines = [
        f"Target: {selection.target.display} / {selection.handedness} ({total_count}/{target_per_handedness})",
        f"Labels: {labels}",
        f"Detected: {detected} [{hand_selection_note}]",
        selection.target.instructions,
        "Space/Enter: capture  |  q/Esc: exit this group",
    ]
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
        f"\nCapturing {selection.target.display} / {selection.handedness}. "
        "Focus the OpenCV preview window, then press Space or Enter to capture."
    )
    print(f"Instruction: {selection.target.instructions}")
    try:
        while selection.existing_count + len(rows) < args.target_per_handedness:
            frame = camera.read()
            if frame is None:
                continue

            hands = tracker.track(frame)
            selected_hand, hand_selection_note = selected_hand_for_target(hands, selection.handedness)
            preview_image = frame.image.copy()
            original_image = frame.image.copy()
            for hand in hands:
                color = (0, 200, 255) if hand is selected_hand else (180, 180, 180)
                draw_hand_preview(preview_image, hand, color)
            draw_status_overlay(
                preview_image,
                selection,
                len(rows),
                args.target_per_handedness,
                selected_hand,
                hand_selection_note,
            )

            cv2.imshow(WINDOW_NAME, preview_image)
            key = cv2.waitKey(1) & 0xFF
            if key in EXIT_KEYS:
                break
            if key not in CAPTURE_KEYS:
                continue
            if selected_hand is None:
                print("No hand detected; capture skipped.")
                continue

            capture_index_for_group = selection.existing_count + len(rows) + 1
            row = capture_row(
                frame,
                original_image,
                selected_hand,
                selection,
                capture_index_for_group,
                hand_selection_note,
                args,
                camera_index,
            )
            rows.append(row)
            print(
                f"Captured {capture_index_for_group}/{args.target_per_handedness}: "
                f"{row['image_path']}"
            )
    finally:
        cv2.destroyWindow(WINDOW_NAME)
    return rows


def run(args: argparse.Namespace) -> int:
    ensure_project_root_on_path()
    from hlgam.capture.camera import CameraSettings, OpenCVCamera
    from hlgam.tracking.hand_tracker import HandTrackerSettings, MediaPipeHandTracker

    if args.remove_group or args.remove_handedness:
        return run_group_removal(args)

    if args.target_per_handedness <= 0:
        raise RuntimeError("--target-per-handedness must be greater than zero")

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
    counts = load_existing_counts(args.output_csv)
    print(f"Writing central CSV rows to {args.output_csv}")
    print(f"Writing captured images under {args.image_root}")

    with OpenCVCamera(camera_settings) as camera, MediaPipeHandTracker(tracker_settings) as tracker:
        while True:
            selection = prompt_for_capture_selection(counts, args.target_per_handedness)
            if selection is None:
                break
            rows = capture_group(camera, tracker, selection, args, camera_index)
            append_rows(args.output_csv, rows)
            if rows:
                key = (selection.target.name, selection.handedness)
                counts[key] = counts.get(key, 0) + len(rows)
                print(f"Stored {len(rows)} rows in {args.output_csv}")
            else:
                print("No rows captured for this group.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = [
        "--remove-group",
        "point_index_down_left",
        "--remove-handedness",
        "left hand"
    ]
    # data/static_poses/images/two_finger_point/right/c0e44fd1d3eb47ff9bfcc980e51d79f1.jpg
    argv = [
        "--remove-group",
        "two_finger_point",
        "--remove-handedness",
        "right hand"
    ]
    argv = None
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Capture interrupted.", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
