from __future__ import annotations

import math

from hlgam.gestures.air_draw import AirDrawConfig, AirDrawRecognizer
from hlgam.gestures.engine import GestureEngine, GestureEngineConfig
from hlgam.gestures.flick import FlickConfig, FlickRecognizer
from hlgam.gestures.pinch_drag import PinchDragConfig, PinchDragRecognizer
from hlgam.gestures.pose import (
    FingerCountConfig,
    FingerCountRecognizer,
    FingerPointingConfig,
    FingerPointingRecognizer,
    FlatHandConfig,
    FlatHandRecognizer,
    StaticPoseConfig,
    StaticPoseRecognizer,
)
from hlgam.gestures.rotation import RotationConfig, RotationRecognizer
from hlgam.gestures.scroll import ScrollConfig, ScrollRecognizer
from hlgam.gestures.swipe import SwipeMotionConfig, SwipeMotionRecognizer
from hlgam.gestures.two_hand import TwoHandGestureConfig, TwoHandGestureRecognizer
from hlgam.gestures.zoom import ZoomConfig, ZoomRecognizer
from hlgam.models import HandLandmarks, Point3D
from hlgam.processing.feature_extractor import FeatureExtractor
from hlgam.processing.normalization import HandNormalizer
from tests.helpers import synthetic_hand


def features_for_sequence():
    normalizer = HandNormalizer()
    extractor = FeatureExtractor()

    def build(timestamp: float, normalized_pinch_distance: float = 0.08, offset_x: float = 0.0, offset_y: float = 0.0):
        raw = synthetic_hand(timestamp, normalized_pinch_distance, offset_x, offset_y)
        return extractor.extract(normalizer.normalize(raw), raw_landmarks=raw)

    return build


def pose_features_for_sequence():
    normalizer = HandNormalizer()
    extractor = FeatureExtractor()

    def build(
        timestamp: float,
        extended: set[str],
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        thumb_direction: str = "side",
        index_direction: str = "up",
        pinch: bool = False,
        spread: bool = False,
    ):
        raw = synthetic_pose_hand(
            timestamp, extended, offset_x, offset_y, thumb_direction, index_direction, pinch, spread
        )
        return extractor.extract(normalizer.normalize(raw), raw_landmarks=raw)

    return build


def orientation_features_for_sequence():
    normalizer = HandNormalizer()
    extractor = FeatureExtractor()

    def build(timestamp: float, angle_radians: float):
        raw = synthetic_orientation_hand(timestamp, angle_radians)
        return extractor.extract(normalizer.normalize(raw), raw_landmarks=raw)

    return build


def synthetic_pose_hand(
    timestamp: float,
    extended: set[str],
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    thumb_direction: str = "side",
    index_direction: str = "up",
    pinch: bool = False,
    spread: bool = False,
) -> HandLandmarks:
    points = [Point3D(offset_x, offset_y, 0.0) for _ in range(21)]
    points[0] = Point3D(offset_x, offset_y, 0.0)
    points[1] = Point3D(offset_x + 0.02, offset_y - 0.01, 0.0)
    points[2] = Point3D(offset_x + 0.05, offset_y - 0.02, 0.0)
    points[3] = Point3D(offset_x + 0.07, offset_y - 0.02, 0.0)
    points[4] = Point3D(offset_x + 0.025, offset_y - 0.01, 0.0)
    if "thumb" in extended:
        if thumb_direction == "up":
            points[3] = Point3D(offset_x, offset_y - 0.07, 0.0)
            points[4] = Point3D(offset_x, offset_y - 0.14, 0.0)
        elif thumb_direction == "down":
            points[3] = Point3D(offset_x, offset_y + 0.07, 0.0)
            points[4] = Point3D(offset_x, offset_y + 0.14, 0.0)
        else:
            points[3] = Point3D(offset_x + 0.09, offset_y - 0.02, 0.0)
            points[4] = Point3D(offset_x + 0.14, offset_y - 0.02, 0.0)

    index_x = -0.06 if spread else -0.04
    middle_x = 0.06 if spread else 0.0
    set_test_finger(points, "index" in extended, 5, 6, 7, 8, offset_x + index_x, offset_y)
    set_test_finger(points, "middle" in extended, 9, 10, 11, 12, offset_x + middle_x, offset_y)
    set_test_finger(points, "ring" in extended, 13, 14, 15, 16, offset_x + 0.04, offset_y)
    set_test_finger(points, "pinky" in extended, 17, 18, 19, 20, offset_x + 0.08, offset_y)
    if "index" in extended:
        set_test_index_direction(points, offset_x, offset_y, index_direction)
    if pinch:
        points[4] = Point3D(offset_x + 0.03, offset_y - 0.12, 0.0)
        points[8] = Point3D(offset_x + 0.034, offset_y - 0.12, 0.0)
    return HandLandmarks(tuple(points), timestamp=timestamp, handedness="right")


def synthetic_orientation_hand(timestamp: float, angle_radians: float) -> HandLandmarks:
    points = [Point3D(0.0, 0.0, 0.0) for _ in range(21)]
    points[9] = Point3D(math.cos(angle_radians) * 0.12, math.sin(angle_radians) * 0.12, 0.0)
    return HandLandmarks(tuple(points), timestamp=timestamp, handedness="right")


def set_test_finger(
    points: list[Point3D],
    extended: bool,
    mcp: int,
    pip: int,
    dip: int,
    tip: int,
    x: float,
    offset_y: float,
) -> None:
    points[mcp] = Point3D(x, offset_y - 0.10, 0.0)
    if extended:
        points[pip] = Point3D(x, offset_y - 0.16, 0.0)
        points[dip] = Point3D(x, offset_y - 0.21, 0.0)
        points[tip] = Point3D(x, offset_y - 0.27, 0.0)
        return
    points[pip] = Point3D(x, offset_y - 0.12, 0.0)
    points[dip] = Point3D(x, offset_y - 0.07, 0.0)
    points[tip] = Point3D(x, offset_y - 0.04, 0.0)


def set_test_index_direction(points: list[Point3D], offset_x: float, offset_y: float, direction: str) -> None:
    vectors = {
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
        "up": (0.0, -1.0),
        "down": (0.0, 1.0),
        "up_left": (-1.0, -1.0),
        "up_right": (1.0, -1.0),
        "down_left": (-1.0, 1.0),
        "down_right": (1.0, 1.0),
    }
    dx, dy = vectors[direction]
    length = 0.18
    if dx and dy:
        length = length / 2**0.5
    base = Point3D(offset_x, offset_y - 0.10, 0.0)
    points[5] = base
    points[6] = Point3D(base.x + dx * length * 0.35, base.y + dy * length * 0.35, 0.0)
    points[7] = Point3D(base.x + dx * length * 0.70, base.y + dy * length * 0.70, 0.0)
    points[8] = Point3D(base.x + dx * length, base.y + dy * length, 0.0)


def test_pinch_recognizer_emits_lifecycle_events_after_hold() -> None:
    build = features_for_sequence()
    recognizer = PinchDragRecognizer(
        PinchDragConfig(
            distance_threshold=0.10,
            release_threshold=0.16,
            hold_ms=100,
            drag_pinch_threshold=0.0,
        )
    )

    assert recognizer.process(build(0.0, 0.08)) == []
    start_events = recognizer.process(build(0.11, 0.08))
    hold_events = recognizer.process(build(0.18, 0.08))
    end_events = recognizer.process(build(0.25, 0.24))

    assert [event.gesture for event in start_events] == ["PINCH_START"]
    assert [event.gesture for event in hold_events] == ["PINCH_HOLD"]
    assert [event.gesture for event in end_events] == ["PINCH_END"]


def test_drag_recognizer_requires_pinch_hold_then_tracks_movement() -> None:
    build = features_for_sequence()
    recognizer = PinchDragRecognizer(
        PinchDragConfig(
            distance_threshold=0.0,
            drag_pinch_threshold=0.10,
            drag_release_threshold=0.16,
            drag_activation_hold_ms=100,
        )
    )

    assert recognizer.process(build(0.0, 0.08, 0.0)) == []
    start_events = recognizer.process(build(0.11, 0.08, 0.0))
    move_events = recognizer.process(build(0.18, 0.08, 0.03))
    end_events = recognizer.process(build(0.25, 0.24, 0.03))

    assert [event.gesture for event in start_events] == ["DRAG_START"]
    assert [event.gesture for event in move_events] == ["DRAG_MOVE"]
    assert [event.gesture for event in end_events] == ["DRAG_END"]


def test_swipe_motion_recognizer_emits_cardinal_swipe() -> None:
    build = features_for_sequence()
    recognizer = SwipeMotionRecognizer(SwipeMotionConfig(min_velocity=0.65, min_distance=0.12, max_duration_ms=500))

    assert recognizer.process(build(0.0, offset_x=0.0)) == []
    assert recognizer.process(build(0.05, offset_x=0.08)) == []
    events = recognizer.process(build(0.10, offset_x=0.23))

    assert [event.gesture for event in events] == ["SWIPE_RIGHT"]


def test_swipe_motion_recognizer_emits_diagonal_swipe() -> None:
    build = features_for_sequence()
    recognizer = SwipeMotionRecognizer(
        SwipeMotionConfig(
            min_velocity=0.65,
            min_distance=0.12,
            diagonal_swipe_min_distance=0.05,
            max_duration_ms=500,
            cooldown_ms=0,
        )
    )

    assert recognizer.process(build(0.0, offset_x=0.0, offset_y=0.0)) == []
    recognizer.process(build(0.05, offset_x=0.05, offset_y=-0.05))
    events = recognizer.process(build(0.10, offset_x=0.12, offset_y=-0.12))

    assert [event.gesture for event in events] == ["DIAGONAL_SWIPE_UP_RIGHT"]


def test_zoom_recognizer_uses_single_hand_pinch_expansion() -> None:
    build = features_for_sequence()
    recognizer = ZoomRecognizer(
        ZoomConfig(min_reference_distance=0.04, min_delta=0.08, min_duration_ms=100, max_duration_ms=700)
    )

    assert recognizer.process(build(0.0, 0.08)) == []
    assert recognizer.process(build(0.05, 0.18)) == []
    events = recognizer.process(build(0.20, 0.18))

    assert [event.gesture for event in events] == ["ZOOM_IN"]
    assert events[0].metadata["zoom_duration_ms"] >= 100


def test_zoom_recognizer_ignores_short_delta_flukes() -> None:
    build = features_for_sequence()
    recognizer = ZoomRecognizer(
        ZoomConfig(min_reference_distance=0.04, min_delta=0.08, min_duration_ms=100, max_duration_ms=700)
    )

    assert recognizer.process(build(0.0, 0.08)) == []
    assert recognizer.process(build(0.03, 0.18)) == []
    assert recognizer.process(build(0.07, 0.08)) == []


def test_zoom_recognizer_emits_zoom_out_after_min_duration() -> None:
    build = features_for_sequence()
    recognizer = ZoomRecognizer(
        ZoomConfig(min_reference_distance=0.04, min_delta=0.08, min_duration_ms=100, max_duration_ms=700)
    )

    assert recognizer.process(build(0.0, 0.18)) == []
    assert recognizer.process(build(0.05, 0.08)) == []
    events = recognizer.process(build(0.20, 0.08))

    assert [event.gesture for event in events] == ["ZOOM_OUT"]


def test_static_pose_recognizer_detects_open_palm_only_from_static_group() -> None:
    build = pose_features_for_sequence()
    recognizer = StaticPoseRecognizer(StaticPoseConfig(min_hold_ms=0, cooldown_ms=0))

    events = recognizer.process(build(0.0, {"thumb", "index", "middle", "ring", "pinky"}))

    gestures = {event.gesture for event in events}
    assert "OPEN_PALM" in gestures
    assert "FINGER_COUNT_5" not in gestures
    assert "FLAT_HAND_UP" not in gestures


def test_finger_count_recognizer_detects_counts_independently() -> None:
    build = pose_features_for_sequence()
    recognizer = FingerCountRecognizer(FingerCountConfig(min_hold_ms=0, cooldown_ms=0))

    events = recognizer.process(build(0.0, {"thumb", "index", "middle"}))

    assert [event.gesture for event in events] == ["FINGER_COUNT_3"]
    assert events[0].metadata["source"] == "finger_count"


def test_flat_hand_recognizer_detects_directions_independently() -> None:
    build = pose_features_for_sequence()
    recognizer = FlatHandRecognizer(FlatHandConfig(min_hold_ms=0, cooldown_ms=0))

    events = recognizer.process(build(0.0, {"thumb", "index", "middle", "ring", "pinky"}))

    assert [event.gesture for event in events] == ["FLAT_HAND_UP"]
    assert events[0].metadata["source"] == "flat_hand"


def test_finger_pointing_recognizer_detects_eight_directions() -> None:
    build = pose_features_for_sequence()
    recognizer = FingerPointingRecognizer(FingerPointingConfig(min_hold_ms=0, cooldown_ms=0))
    expected = {
        "left": "FINGER_POINT_LEFT",
        "right": "FINGER_POINT_RIGHT",
        "up": "FINGER_POINT_UP",
        "down": "FINGER_POINT_DOWN",
        "up_left": "FINGER_POINT_UP_LEFT",
        "up_right": "FINGER_POINT_UP_RIGHT",
        "down_left": "FINGER_POINT_DOWN_LEFT",
        "down_right": "FINGER_POINT_DOWN_RIGHT",
    }

    for index, (direction, gesture) in enumerate(expected.items()):
        events = recognizer.process(build(index / 10, {"index"}, index_direction=direction))

        assert [event.gesture for event in events] == [gesture]
        assert events[0].metadata["source"] == "finger_pointing"
        assert events[0].metadata["direction"] == gesture.removeprefix("FINGER_POINT_")


def test_gesture_engine_toggles_static_pose_subgroups_independently() -> None:
    build = pose_features_for_sequence()
    engine = GestureEngine(
        GestureEngineConfig(
            pose=StaticPoseConfig(enabled=False, min_hold_ms=0, cooldown_ms=0),
            finger_count=FingerCountConfig(enabled=True, min_hold_ms=0, cooldown_ms=0),
            finger_pointing=FingerPointingConfig(enabled=False, min_hold_ms=0, cooldown_ms=0),
            flat_hand=FlatHandConfig(enabled=False, min_hold_ms=0, cooldown_ms=0),
        )
    )

    events = engine.process(build(0.0, {"thumb", "index", "middle", "ring", "pinky"}))
    gestures = {event.gesture for event in events}

    assert "FINGER_COUNT_5" in gestures
    assert "OPEN_PALM" not in gestures
    assert "FLAT_HAND_UP" not in gestures


def test_gesture_engine_toggles_finger_pointing_independently() -> None:
    build = pose_features_for_sequence()
    engine = GestureEngine(
        GestureEngineConfig(
            pose=StaticPoseConfig(enabled=False, min_hold_ms=0, cooldown_ms=0),
            finger_count=FingerCountConfig(enabled=False, min_hold_ms=0, cooldown_ms=0),
            finger_pointing=FingerPointingConfig(enabled=True, min_hold_ms=0, cooldown_ms=0),
            flat_hand=FlatHandConfig(enabled=False, min_hold_ms=0, cooldown_ms=0),
        )
    )

    events = engine.process(build(0.0, {"index"}, index_direction="right"))
    gestures = {event.gesture for event in events}

    assert "FINGER_POINT_RIGHT" in gestures
    assert "POINT_INDEX" not in gestures
    assert "FINGER_COUNT_1" not in gestures


def test_static_pose_recognizer_detects_specific_hand_shapes() -> None:
    build = pose_features_for_sequence()
    recognizer = StaticPoseRecognizer(StaticPoseConfig(min_hold_ms=0, cooldown_ms=0))

    point_events = recognizer.process(build(0.0, {"index"}))
    l_shape_events = recognizer.process(build(0.1, {"thumb", "index"}))
    ok_events = recognizer.process(build(0.2, {"index", "middle", "ring", "pinky"}, pinch=True))
    thumb_events = recognizer.process(build(0.3, {"thumb"}, thumb_direction="up"))
    peace_events = recognizer.process(build(0.4, {"index", "middle"}, spread=True))

    assert "POINT_INDEX" in {event.gesture for event in point_events}
    assert "L_SHAPE" in {event.gesture for event in l_shape_events}
    assert "OK_SIGN" in {event.gesture for event in ok_events}
    assert "THUMBS_UP" in {event.gesture for event in thumb_events}
    assert "PEACE_SIGN" in {event.gesture for event in peace_events}


def test_flick_recognizer_detects_short_motion() -> None:
    build = features_for_sequence()
    recognizer = FlickRecognizer(FlickConfig(min_velocity=0.65, min_distance=0.04, max_duration_ms=180, cooldown_ms=0))

    assert recognizer.process(build(0.0, offset_x=0.0, offset_y=0.0)) == []
    events = recognizer.process(build(0.05, offset_x=0.08, offset_y=0.0))

    assert [event.gesture for event in events] == ["FLICK_RIGHT"]


def test_rotation_recognizer_detects_circle_path() -> None:
    build = pose_features_for_sequence()
    recognizer = RotationRecognizer(
        RotationConfig(circle_min_radius=0.04, circle_min_arc_degrees=300.0, cooldown_ms=0)
    )
    events = []
    for index in range(7):
        angle = math.tau * (index / 6)
        events = recognizer.process(
            build(index / 10, {"index"}, offset_x=math.cos(angle) * 0.08, offset_y=math.sin(angle) * 0.08)
        )

    assert [event.gesture for event in events] == ["CIRCLE_CLOCKWISE"]


def test_rotation_recognizer_detects_hand_orientation_rotation() -> None:
    build = orientation_features_for_sequence()
    recognizer = RotationRecognizer(RotationConfig(rotate_min_degrees=45.0, cooldown_ms=0))

    assert recognizer.process(build(0.0, 0.0)) == []
    events = recognizer.process(build(0.1, math.radians(90.0)))

    assert [event.gesture for event in events] == ["ROTATE_CLOCKWISE"]


def test_scroll_recognizer_detects_cardinal_two_finger_scrolls() -> None:
    build = pose_features_for_sequence()
    cases = [
        (0.0, 0.09, 0.0, "SCROLL_RIGHT"),
        (1.0, -0.09, 0.0, "SCROLL_LEFT"),
        (2.0, 0.0, -0.09, "SCROLL_UP"),
        (3.0, 0.0, 0.09, "SCROLL_DOWN"),
    ]

    for timestamp, offset_x, offset_y, gesture in cases:
        recognizer = ScrollRecognizer(ScrollConfig(min_distance=0.05, cooldown_ms=0))

        assert recognizer.process(build(timestamp, {"index", "middle"})) == []
        events = recognizer.process(build(timestamp + 0.1, {"index", "middle"}, offset_x, offset_y))

        assert [event.gesture for event in events] == [gesture]
        assert events[0].metadata["direction"] == gesture.removeprefix("SCROLL_")


def test_pinch_drag_recognizer_detects_pinch_tap() -> None:
    build = features_for_sequence()
    recognizer = PinchDragRecognizer(
        PinchDragConfig(
            distance_threshold=0.0,
            drag_pinch_threshold=0.0,
            pinch_tap_min_duration_ms=0,
            pinch_tap_max_duration_ms=120,
        )
    )

    assert recognizer.process(build(0.0, 0.08)) == []
    events = recognizer.process(build(0.06, 0.24))

    assert "PINCH_TAP" in {event.gesture for event in events}


def test_air_draw_recognizer_streams_lifecycle() -> None:
    build = pose_features_for_sequence()
    recognizer = AirDrawRecognizer(AirDrawConfig(min_distance=0.005))

    start_events = recognizer.process(build(0.0, {"index"}, offset_x=0.0))
    move_events = recognizer.process(build(0.1, {"index"}, offset_x=0.03))
    end_events = recognizer.process(build(0.2, {"thumb", "index", "middle", "ring", "pinky"}, offset_x=0.03))

    assert "AIR_DRAW_START" in {event.gesture for event in start_events}
    assert "AIR_DRAW_MOVE" in {event.gesture for event in move_events}
    assert "AIR_DRAW_END" in {event.gesture for event in end_events}


def test_two_hand_recognizer_detects_pair_zoom() -> None:
    build = pose_features_for_sequence()
    recognizer = TwoHandGestureRecognizer(TwoHandGestureConfig(zoom_min_delta=0.08, cooldown_ms=0))

    assert recognizer.process_many([
        build(0.0, {"thumb", "index", "middle", "ring", "pinky"}, offset_x=-0.20),
        build(0.0, {"thumb", "index", "middle", "ring", "pinky"}, offset_x=0.20),
    ]) == []
    events = recognizer.process_many([
        build(0.1, {"thumb", "index", "middle", "ring", "pinky"}, offset_x=-0.30),
        build(0.1, {"thumb", "index", "middle", "ring", "pinky"}, offset_x=0.30),
    ])

    assert "TWO_HAND_ZOOM_IN" in {event.gesture for event in events}
