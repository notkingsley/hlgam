#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "e2e_camera_events.jsonl"

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


@dataclass
class RunStats:
    frames: int = 0
    dropped_frames: int = 0
    frames_with_hands: int = 0
    events: int = 0
    started_at: float = 0.0

    def fps(self) -> float:
        elapsed = max(time.perf_counter() - self.started_at, 1e-6)
        return self.frames / elapsed


class StopFlag:
    def __init__(self) -> None:
        self.requested = False

    def request_stop(self, signum: int, frame) -> None:
        self.requested = True


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
        fps = f"{self.fps:.1f}fps" if self.fps else "unknown fps"
        return f"camera index {self.index} ({resolution}, {fps}, {status})"


class TerminalAndFileEventSink:
    def __init__(self, log_file: Path, stats: RunStats, truncate: bool = False) -> None:
        ensure_project_root_on_path()
        from hlgam.evaluation.logger import GestureEventLogger

        self.log_file = log_file
        self.stats = stats
        if truncate:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text("", encoding="utf-8")
        self.logger = GestureEventLogger(log_file)

    def handle(self, event: Any) -> None:
        self.stats.events += 1
        self.logger.write(event)
        print(event.to_json(), flush=True)


def ensure_project_root_on_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open the camera, run HLGAM detection, and log detected gesture events."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to a YAML or JSON HLGAM config file.")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help="JSONL file for detected gesture events.",
    )
    parser.add_argument("--truncate-log", action="store_true", help="Clear the log file before writing new events.")
    parser.add_argument("--camera-index", type=int, help="Use a specific camera index and skip camera selection.")
    parser.add_argument(
        "--camera-scan-limit",
        type=int,
        default=1,
        help="Highest camera index to probe when --camera-index is not supplied.",
    )
    parser.add_argument(
        "--no-camera-prompt",
        action="store_true",
        help=(
            "Do not prompt for camera selection; use the config camera if detected, "
            "otherwise the first detected camera."
        ),
    )
    parser.add_argument("--width", type=int, help="Override camera frame width.")
    parser.add_argument("--height", type=int, help="Override camera frame height.")
    parser.add_argument("--fps", type=int, help="Override camera target FPS.")
    mirror_group = parser.add_mutually_exclusive_group()
    mirror_group.add_argument(
        "--mirror",
        dest="mirror",
        action="store_true",
        default=None,
        help="Mirror camera input horizontally before tracking and preview.",
    )
    mirror_group.add_argument(
        "--no-mirror",
        dest="mirror",
        action="store_false",
        help="Do not mirror camera input horizontally.",
    )
    parser.add_argument("--max-frames", type=int, help="Stop after processing this many frames.")
    parser.add_argument("--preview", action="store_true", help="Show a live OpenCV preview. Press q to stop.")
    parser.add_argument("--status-interval", type=float, default=2.0, help="Seconds between status updates.")
    parser.add_argument("--min-detection-confidence", type=float, default=0.6, help="MediaPipe detection threshold.")
    parser.add_argument("--min-tracking-confidence", type=float, default=0.6, help="MediaPipe tracking threshold.")
    parser.add_argument("--model-complexity", type=int, choices=(0, 1), default=1, help="MediaPipe model complexity.")
    return parser.parse_args(argv)


def load_runtime_config(path: Path) -> Any:
    ensure_project_root_on_path()
    from hlgam.config import load_config

    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    return load_config(path)


def build_camera_settings(config: Any, args: argparse.Namespace, camera_index: int) -> Any:
    ensure_project_root_on_path()
    from hlgam.capture.camera import CameraSettings

    return CameraSettings(
        index=camera_index,
        width=args.width if args.width is not None else config.camera.width,
        height=args.height if args.height is not None else config.camera.height,
        fps=args.fps if args.fps is not None else config.camera.fps,
        mirror=args.mirror if args.mirror is not None else config.camera.mirror,
    )


def detect_present_cameras(scan_limit: int, logger: logging.Logger) -> list[CameraCandidate]:
    import cv2

    upper_bound = max(0, scan_limit)
    candidates: list[CameraCandidate] = []
    logger.info("scanning camera indices 0 through %s", upper_bound)

    for index in range(upper_bound + 1):
        capture = cv2.VideoCapture(index)
        try:
            if not capture.isOpened():
                continue

            readable, frame = capture.read()
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if readable and frame is not None:
                height, width = frame.shape[:2]

            candidates.append(
                CameraCandidate(
                    index=index,
                    width=width,
                    height=height,
                    fps=fps,
                    readable=readable,
                )
            )
        finally:
            capture.release()

    return candidates


def select_default_camera_index(candidates: list[CameraCandidate], preferred_index: int) -> int:
    for candidate in candidates:
        if candidate.index == preferred_index:
            return candidate.index
    for candidate in candidates:
        if candidate.readable:
            return candidate.index
    return candidates[0].index


def prompt_for_camera(candidates: list[CameraCandidate], default_index: int) -> int:
    candidate_indices = {candidate.index for candidate in candidates}
    print("\nDetected cameras:", file=sys.stderr)
    for candidate in candidates:
        marker = "*" if candidate.index == default_index else " "
        print(f"  {marker} {candidate.description()}", file=sys.stderr)
    print("Enter a camera index from the list, press Enter for the default, or type q to quit.", file=sys.stderr)

    while True:
        selected = input(f"Select camera index [{default_index}]: ").strip().lower()
        if selected == "":
            return default_index
        if selected in {"q", "quit", "exit"}:
            raise RuntimeError("camera selection cancelled")
        try:
            index = int(selected)
        except ValueError:
            print("Please enter a numeric camera index, or q to quit.", file=sys.stderr)
            continue
        if index in candidate_indices:
            return index
        print("That camera index was not detected. Choose one from the list.", file=sys.stderr)


def resolve_camera_index(config: Any, args: argparse.Namespace, logger: logging.Logger) -> int:
    if args.camera_index is not None:
        logger.info("using camera index %s from --camera-index; skipping camera scan", args.camera_index)
        return args.camera_index

    candidates = detect_present_cameras(args.camera_scan_limit, logger)
    if not candidates:
        logger.warning("no cameras detected; falling back to configured camera index %s", config.camera.index)
        return config.camera.index

    default_index = select_default_camera_index(candidates, config.camera.index)
    if args.no_camera_prompt or not sys.stdin.isatty():
        selected = next(candidate for candidate in candidates if candidate.index == default_index)
        logger.info("selected %s", selected.description())
        return default_index

    selected_index = prompt_for_camera(candidates, default_index)
    selected = next(candidate for candidate in candidates if candidate.index == selected_index)
    logger.info("selected %s", selected.description())
    return selected_index


def build_tracker_settings(args: argparse.Namespace) -> Any:
    ensure_project_root_on_path()
    from hlgam.tracking.hand_tracker import HandTrackerSettings

    return HandTrackerSettings(
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        model_complexity=args.model_complexity,
    )


def configure_status_logger() -> logging.Logger:
    logger = logging.getLogger("hlgam.e2e")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    return logger


def draw_hand_preview(image: Any, hand: Any) -> None:
    import cv2

    height, width = image.shape[:2]
    points = [(int(point.x * width), int(point.y * height)) for point in hand.landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(image, points[start], points[end], (0, 180, 255), 2)
    for point in points:
        cv2.circle(image, point, 4, (0, 255, 0), -1)


def show_preview(image) -> bool:
    import cv2

    cv2.imshow("HLGAM E2E Camera Test", image)
    key = cv2.waitKey(1) & 0xFF
    return key not in {ord("q"), 27}


def close_preview() -> None:
    try:
        import cv2
    except ImportError:
        return
    cv2.destroyAllWindows()


def log_status(logger: logging.Logger, stats: RunStats) -> None:
    logger.info(
        "frames=%s hands=%s events=%s dropped=%s fps=%.1f",
        stats.frames,
        stats.frames_with_hands,
        stats.events,
        stats.dropped_frames,
        stats.fps(),
    )


def run(args: argparse.Namespace) -> int:
    ensure_project_root_on_path()
    from hlgam.api.framework import GestureFramework
    from hlgam.capture.camera import OpenCVCamera
    from hlgam.tracking.hand_tracker import MediaPipeHandTracker

    logger = configure_status_logger()
    config = load_runtime_config(args.config)
    camera_index = resolve_camera_index(config, args, logger)
    camera_settings = build_camera_settings(config, args, camera_index)
    tracker_settings = build_tracker_settings(args)
    framework = GestureFramework(config)
    stats = RunStats(started_at=time.perf_counter())
    sink = TerminalAndFileEventSink(args.log_file, stats, truncate=args.truncate_log)
    framework.on_any(sink.handle)

    stop_flag = StopFlag()
    signal.signal(signal.SIGINT, stop_flag.request_stop)
    signal.signal(signal.SIGTERM, stop_flag.request_stop)

    logger.info("starting HLGAM E2E camera test")
    logger.info(
        "camera index=%s resolution=%sx%s fps=%s mirror=%s",
        camera_settings.index,
        camera_settings.width,
        camera_settings.height,
        camera_settings.fps,
        camera_settings.mirror,
    )
    logger.info("writing gesture events to %s", args.log_file)
    logger.info("printing detected gesture events as JSON lines to stdout")

    next_status_at = time.perf_counter() + args.status_interval
    try:
        with OpenCVCamera(camera_settings) as camera, MediaPipeHandTracker(tracker_settings) as tracker:
            while not stop_flag.requested:
                if args.max_frames is not None and stats.frames >= args.max_frames:
                    break

                frame = camera.read()
                if frame is None:
                    stats.dropped_frames += 1
                    continue

                stats.frames += 1
                hands = tracker.track(frame)
                if hands:
                    stats.frames_with_hands += 1
                    framework.process_landmarks(hands[0])
                    if args.preview:
                        draw_hand_preview(frame.image, hands[0])
                else:
                    framework.reset()

                if args.preview and not show_preview(frame.image):
                    stop_flag.requested = True

                now = time.perf_counter()
                if now >= next_status_at:
                    log_status(logger, stats)
                    next_status_at = now + args.status_interval
    finally:
        if args.preview:
            close_preview()

    log_status(logger, stats)
    logger.info("HLGAM E2E camera test stopped")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = [
        "--mirror",
        "--preview",
    ]
    # argv = None
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        logger = configure_status_logger()
        logger.info("HLGAM E2E camera test interrupted")
        return 130
    except (FileNotFoundError, RuntimeError) as exc:
        logger = configure_status_logger()
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
