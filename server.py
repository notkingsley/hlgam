#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "server_events.jsonl"

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
class ServerStats:
    frames: int = 0
    dropped_frames: int = 0
    frames_with_hands: int = 0
    events: int = 0
    started_at: float = 0.0

    def fps(self) -> float:
        elapsed = max(time.perf_counter() - self.started_at, 1e-6)
        return self.frames / elapsed


def ensure_project_root_on_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start an HLGAM camera/synthetic gesture server for the React Router UI."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to a YAML or JSON HLGAM config file.")
    parser.add_argument("--host", help="WebSocket host. Defaults to broker.websocket.host in the config.")
    parser.add_argument("--port", type=int, help="WebSocket port. Defaults to broker.websocket.port in the config.")
    parser.add_argument(
        "--source",
        choices=("camera", "synthetic"),
        default="camera",
        help="Use a live webcam or generate synthetic demo gestures.",
    )

    jwt_group = parser.add_mutually_exclusive_group()
    jwt_group.add_argument("--require-jwt", dest="require_jwt", action="store_true", help="Require JWT auth.")
    jwt_group.add_argument(
        "--no-jwt",
        dest="require_jwt",
        action="store_false",
        help="Disable JWT auth for local development with the UI.",
    )
    parser.set_defaults(require_jwt=None)
    parser.add_argument("--jwt-env-file", help="Path to .env file containing HLGAM_WS_JWT_* settings.")
    parser.add_argument("--jwt-auth-timeout", type=float, help="Seconds to wait for first-message JWT auth.")

    parser.add_argument("--camera-index", type=int, help="Open this camera index instead of config.camera.index.")
    parser.add_argument("--width", type=int, help="Override camera frame width.")
    parser.add_argument("--height", type=int, help="Override camera frame height.")
    parser.add_argument("--fps", type=int, help="Override camera target FPS.")
    parser.add_argument("--max-hands", type=int, default=2, help="Maximum MediaPipe hands to track.")
    mirror_group = parser.add_mutually_exclusive_group()
    mirror_group.add_argument("--mirror", dest="mirror", action="store_true", default=None, help="Mirror camera input.")
    mirror_group.add_argument("--no-mirror", dest="mirror", action="store_false", help="Do not mirror camera input.")
    parser.add_argument("--min-detection-confidence", type=float, default=0.6, help="MediaPipe detection threshold.")
    parser.add_argument("--min-tracking-confidence", type=float, default=0.6, help="MediaPipe tracking threshold.")
    parser.add_argument("--model-complexity", type=int, choices=(0, 1), default=1, help="MediaPipe model complexity.")

    parser.add_argument("--preview", action="store_true", help="Show an OpenCV preview window for camera source.")
    parser.add_argument("--synthetic-interval", type=float, default=0.08, help="Seconds between synthetic frames.")
    parser.add_argument("--status-interval", type=float, default=2.0, help="Seconds between status log lines.")
    parser.add_argument("--max-frames", type=int, help="Stop after this many source frames.")
    parser.add_argument("--print-events", action="store_true", help="Print locally recognized events as JSON lines.")
    parser.add_argument("--log-file", type=Path, help="Optional JSONL log file for locally recognized events.")
    parser.add_argument("--truncate-log", action="store_true", help="Clear --log-file before writing.")
    return parser.parse_args(argv)


def configure_logger() -> logging.Logger:
    logger = logging.getLogger("hlgam.server")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    return logger


def load_runtime_config(path: Path) -> Any:
    ensure_project_root_on_path()
    from hlgam.config import load_config

    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    return load_config(path)


def build_camera_settings(config: Any, args: argparse.Namespace) -> Any:
    ensure_project_root_on_path()
    from hlgam.capture.camera import CameraSettings

    return CameraSettings(
        index=args.camera_index if args.camera_index is not None else config.camera.index,
        width=args.width if args.width is not None else config.camera.width,
        height=args.height if args.height is not None else config.camera.height,
        fps=args.fps if args.fps is not None else config.camera.fps,
        mirror=args.mirror if args.mirror is not None else config.camera.mirror,
    )


def build_tracker_settings(args: argparse.Namespace) -> Any:
    ensure_project_root_on_path()
    from hlgam.tracking.hand_tracker import HandTrackerSettings

    return HandTrackerSettings(
        max_num_hands=max(1, args.max_hands),
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        model_complexity=args.model_complexity,
    )


def build_websocket_broker(config: Any, args: argparse.Namespace) -> Any:
    ensure_project_root_on_path()
    from hlgam.broker.websocket_broker import WebSocketGestureBroker
    from hlgam.gestures.engine import GestureEngineConfig

    websocket = config.broker.websocket
    require_jwt = websocket.require_jwt if args.require_jwt is None else args.require_jwt
    return WebSocketGestureBroker(
        host=args.host or websocket.host,
        port=args.port if args.port is not None else websocket.port,
        require_jwt=require_jwt,
        jwt_env_file=args.jwt_env_file or websocket.jwt_env_file,
        jwt_auth_timeout_seconds=(
            args.jwt_auth_timeout if args.jwt_auth_timeout is not None else websocket.jwt_auth_timeout_seconds
        ),
        default_engine_config=GestureEngineConfig(
            pinch_drag=config.gestures.pinch_drag,
            swipe_motion=config.gestures.swipe_motion,
            zoom=config.gestures.zoom,
            pose=config.gestures.pose,
            finger_count=config.gestures.finger_count,
            finger_pointing=config.gestures.finger_pointing,
            flat_hand=config.gestures.flat_hand,
            flick=config.gestures.flick,
            rotation=config.gestures.rotation,
            scroll=config.gestures.scroll,
            motion=config.gestures.motion,
            air_draw=config.gestures.air_draw,
            two_hand=config.gestures.two_hand,
            recognition=config.recognition,
            gesture_events=config.gesture_events,
            gesture_mapping=config.gesture_mapping,
        ),
        tracker_settings=build_tracker_settings(args),
    )


def attach_local_event_sinks(framework: Any, args: argparse.Namespace, stats: ServerStats) -> None:
    if not args.print_events and args.log_file is None:
        return

    logger = None
    if args.log_file is not None:
        ensure_project_root_on_path()
        from hlgam.evaluation.logger import GestureEventLogger

        log_file = args.log_file or DEFAULT_LOG_FILE
        if args.truncate_log:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text("", encoding="utf-8")
        logger = GestureEventLogger(log_file)

    def handle_event(event: Any) -> None:
        stats.events += 1
        if logger is not None:
            logger.write(event)
        if args.print_events:
            print(event.to_json(), flush=True)

    framework.on_any(handle_event)


async def run_server(args: argparse.Namespace) -> int:
    ensure_project_root_on_path()
    from hlgam.api.framework import GestureFramework

    logger = configure_logger()
    config = load_runtime_config(args.config)
    framework = GestureFramework(config)
    broker = build_websocket_broker(config, args)
    stats = ServerStats(started_at=time.perf_counter())
    attach_local_event_sinks(framework, args, stats)

    stop_event = asyncio.Event()
    install_signal_handlers(stop_event)

    await broker.start()
    logger.info("WebSocket server listening on ws://%s:%s", broker.host, broker.port)
    if broker.require_jwt:
        logger.info("JWT auth is required; paste a valid token into the UI or restart with --no-jwt for local dev")
    else:
        logger.info("JWT auth is disabled; the UI can connect without a token")
    logger.info("source=%s max_hands=%s", args.source, args.max_hands)

    try:
        if args.source == "synthetic":
            await run_synthetic_source(framework, broker, args, stats, stop_event, logger)
        else:
            await run_camera_source(framework, broker, args, stats, stop_event, logger)
    finally:
        await broker.stop()
        close_preview()
        log_status(logger, stats)
        logger.info("HLGAM server stopped")
    return 0


async def run_camera_source(
    framework: Any,
    broker: Any,
    args: argparse.Namespace,
    stats: ServerStats,
    stop_event: asyncio.Event,
    logger: logging.Logger,
) -> None:
    ensure_project_root_on_path()
    from hlgam.capture.camera import OpenCVCamera
    from hlgam.tracking.hand_tracker import MediaPipeHandTracker

    config = load_runtime_config(args.config)
    camera_settings = build_camera_settings(config, args)
    tracker_settings = build_tracker_settings(args)
    logger.info(
        "opening camera index=%s resolution=%sx%s fps=%s mirror=%s",
        camera_settings.index,
        camera_settings.width,
        camera_settings.height,
        camera_settings.fps,
        camera_settings.mirror,
    )
    next_status_at = time.perf_counter() + args.status_interval

    with OpenCVCamera(camera_settings) as camera, MediaPipeHandTracker(tracker_settings) as tracker:
        while not stop_event.is_set():
            if args.max_frames is not None and stats.frames >= args.max_frames:
                break

            frame = camera.read()
            if frame is None:
                stats.dropped_frames += 1
                await asyncio.sleep(0)
                continue

            stats.frames += 1
            hands = tracker.track(frame)
            if hands:
                stats.frames_with_hands += 1
                events = framework.process_hands(hands)
                await broker.process_source_frame("internal_camera", frame, hands)
                stats.events += len(events) if not (args.print_events or args.log_file) else 0
                if args.preview:
                    for hand in hands:
                        draw_hand_preview(frame.image, hand)
            else:
                framework.reset()
                await broker.process_source_frame("internal_camera", frame, [])

            if args.preview and not show_preview(frame.image):
                stop_event.set()

            now = time.perf_counter()
            if now >= next_status_at:
                log_status(logger, stats)
                next_status_at = now + args.status_interval
            await asyncio.sleep(0)


async def run_synthetic_source(
    framework: Any,
    broker: Any,
    args: argparse.Namespace,
    stats: ServerStats,
    stop_event: asyncio.Event,
    logger: logging.Logger,
) -> None:
    logger.info("starting synthetic gesture source")
    next_status_at = time.perf_counter() + args.status_interval
    step = 0
    while not stop_event.is_set():
        if args.max_frames is not None and stats.frames >= args.max_frames:
            break

        timestamp = time.perf_counter()
        hand = synthetic_demo_hand(timestamp, step)
        stats.frames += 1
        stats.frames_with_hands += 1
        events = framework.process_landmarks(hand)
        await broker.process_source_hands("internal_camera", [hand])
        stats.events += len(events) if not (args.print_events or args.log_file) else 0
        step += 1

        now = time.perf_counter()
        if now >= next_status_at:
            log_status(logger, stats)
            next_status_at = now + args.status_interval
        await asyncio.sleep(max(args.synthetic_interval, 0.001))


def synthetic_demo_hand(timestamp: float, step: int) -> Any:
    ensure_project_root_on_path()
    from hlgam.models import HandLandmarks, Point3D

    phase = step % 48
    pinch_distance = 0.008 if phase < 12 else 0.024
    offset_x = 0.16 * min(phase, 16) / 16 if 12 <= phase < 20 else 0.0
    offset_y = -0.12 * min(phase - 20, 12) / 12 if 20 <= phase < 32 else 0.0
    points = [Point3D(offset_x, offset_y, 0.0) for _ in range(21)]
    points[0] = Point3D(offset_x, offset_y, 0.0)
    points[1] = Point3D(offset_x + 0.01, offset_y - 0.02, 0.0)
    points[2] = Point3D(offset_x + 0.015, offset_y - 0.04, 0.0)
    points[3] = Point3D(offset_x + 0.010, offset_y - 0.08, 0.0)
    points[4] = Point3D(offset_x, offset_y - 0.12, 0.0)
    points[5] = Point3D(offset_x + 0.08, offset_y - 0.04, 0.0)
    points[6] = Point3D(offset_x + 0.05, offset_y - 0.10, 0.0)
    points[7] = Point3D(offset_x + 0.04, offset_y - 0.13, 0.0)
    points[8] = Point3D(offset_x + pinch_distance, offset_y - 0.12, 0.0)
    points[9] = Point3D(offset_x, offset_y - 0.10, 0.0)
    points[10] = Point3D(offset_x, offset_y - 0.16, 0.0)
    points[11] = Point3D(offset_x, offset_y - 0.20, 0.0)
    points[12] = Point3D(offset_x, offset_y - 0.24, 0.0)
    points[13] = Point3D(offset_x - 0.07, offset_y - 0.04, 0.0)
    points[14] = Point3D(offset_x - 0.08, offset_y - 0.11, 0.0)
    points[15] = Point3D(offset_x - 0.08, offset_y - 0.16, 0.0)
    points[16] = Point3D(offset_x - 0.08, offset_y - 0.20, 0.0)
    points[17] = Point3D(offset_x - 0.10, offset_y, 0.0)
    points[18] = Point3D(offset_x - 0.12, offset_y - 0.07, 0.0)
    points[19] = Point3D(offset_x - 0.13, offset_y - 0.11, 0.0)
    points[20] = Point3D(offset_x - 0.14, offset_y - 0.15, 0.0)
    return HandLandmarks(tuple(points), timestamp=timestamp, handedness="right")


def draw_hand_preview(image: Any, hand: Any) -> None:
    import cv2

    height, width = image.shape[:2]
    points = [(int(point.x * width), int(point.y * height)) for point in hand.landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(image, points[start], points[end], (0, 180, 255), 2)
    for point in points:
        cv2.circle(image, point, 4, (0, 255, 0), -1)


def show_preview(image: Any) -> bool:
    import cv2

    cv2.imshow("HLGAM WebSocket Server", image)
    key = cv2.waitKey(1) & 0xFF
    return key not in {ord("q"), 27}


def close_preview() -> None:
    try:
        import cv2
    except ImportError:
        return
    cv2.destroyAllWindows()


def log_status(logger: logging.Logger, stats: ServerStats) -> None:
    logger.info(
        "frames=%s hands=%s events=%s dropped=%s fps=%.1f",
        stats.frames,
        stats.frames_with_hands,
        stats.events,
        stats.dropped_frames,
        stats.fps(),
    )


def install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_event.set)
        except NotImplementedError:
            signal.signal(signum, lambda _signum, _frame: stop_event.set())


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(run_server(args))
    except KeyboardInterrupt:
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        configure_logger().error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
