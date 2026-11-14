from __future__ import annotations

import math
from dataclasses import dataclass

from hlgam import landmark_indices as idx
from hlgam.gestures.base import clamp_confidence
from hlgam.models import GestureEvent, GestureFeatures, GesturePhase, Point3D


FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class TimedPoseConfig:
    enabled: bool = True
    min_hold_ms: int = 80
    cooldown_ms: int = 250


@dataclass(frozen=True)
class StaticPoseConfig(TimedPoseConfig):
    pinch_distance_threshold: float = 0.12
    ok_distance_threshold: float = 0.14
    spread_distance_threshold: float = 0.45
    direction_ratio: float = 1.25
    l_shape_min_angle_degrees: float = 55.0
    l_shape_max_angle_degrees: float = 125.0


@dataclass(frozen=True)
class FingerCountConfig(TimedPoseConfig):
    pass


@dataclass(frozen=True)
class FlatHandConfig(TimedPoseConfig):
    direction_ratio: float = 1.25


@dataclass(frozen=True)
class FingerPointingConfig(TimedPoseConfig):
    direction_ratio: float = 1.25
    min_vector_length: float = 0.05


class TimedPoseRecognizer:
    name = "pose"
    metadata_source = "pose"

    def __init__(self, config: TimedPoseConfig) -> None:
        self.config = config
        self._active_since: dict[str, float] = {}
        self._cooldown_until: dict[str, float] = {}

    def process(self, features: GestureFeatures) -> list[GestureEvent]:
        if not self.config.enabled:
            return []

        timestamp = features.timestamp
        candidates = set(self.candidate_names(features))
        for gesture in tuple(self._active_since):
            if gesture not in candidates:
                self._active_since.pop(gesture, None)

        events: list[GestureEvent] = []
        for gesture in sorted(candidates):
            active_since = self._active_since.setdefault(gesture, timestamp)
            held_ms = (timestamp - active_since) * 1000.0
            if held_ms < self.config.min_hold_ms:
                continue
            if timestamp < self._cooldown_until.get(gesture, 0.0):
                continue
            self._cooldown_until[gesture] = timestamp + (self.config.cooldown_ms / 1000.0)
            events.append(self._event(gesture, features, held_ms))
        return events

    def reset(self) -> None:
        self._active_since.clear()
        self._cooldown_until.clear()

    def candidate_names(self, features: GestureFeatures) -> list[str]:
        raise NotImplementedError

    def confidence(self, features: GestureFeatures, gesture: str) -> float:
        return 0.9

    def _event(self, gesture: str, features: GestureFeatures, held_ms: float) -> GestureEvent:
        return GestureEvent(
            gesture=gesture,
            phase=GesturePhase.COMPLETED,
            confidence=self.confidence(features, gesture),
            timestamp=features.timestamp,
            position=features.position_payload(),
            velocity=features.velocity_payload(),
            metadata={
                "held_ms": held_ms,
                "extended_fingers": list(extended_fingers(features)),
                "finger_count": finger_count(features),
                "handedness": features.landmarks.handedness,
                "source": self.metadata_source,
            },
        )


class StaticPoseRecognizer(TimedPoseRecognizer):
    name = "pose"
    metadata_source = "static_pose"

    def __init__(self, config: StaticPoseConfig | None = None) -> None:
        super().__init__(config or StaticPoseConfig())
        self.config: StaticPoseConfig

    def candidate_names(self, features: GestureFeatures) -> list[str]:
        return candidate_pose_names(features, self.config)

    def confidence(self, features: GestureFeatures, gesture: str) -> float:
        return pose_confidence(features, gesture, self.config)


class FingerCountRecognizer(TimedPoseRecognizer):
    name = "finger_count"
    metadata_source = "finger_count"

    def __init__(self, config: FingerCountConfig | None = None) -> None:
        super().__init__(config or FingerCountConfig())

    def candidate_names(self, features: GestureFeatures) -> list[str]:
        count = finger_count(features)
        return [f"FINGER_COUNT_{count}"] if count > 0 else []

    def confidence(self, features: GestureFeatures, gesture: str) -> float:
        return 0.85


class FlatHandRecognizer(TimedPoseRecognizer):
    name = "flat_hand"
    metadata_source = "flat_hand"

    def __init__(self, config: FlatHandConfig | None = None) -> None:
        super().__init__(config or FlatHandConfig())
        self.config: FlatHandConfig

    def candidate_names(self, features: GestureFeatures) -> list[str]:
        if finger_count(features) != 5:
            return []
        direction = flat_hand_direction(features, self.config.direction_ratio)
        return [direction] if direction is not None else []

    def confidence(self, features: GestureFeatures, gesture: str) -> float:
        return 0.8


class FingerPointingRecognizer(TimedPoseRecognizer):
    name = "finger_pointing"
    metadata_source = "finger_pointing"

    def __init__(self, config: FingerPointingConfig | None = None) -> None:
        super().__init__(config or FingerPointingConfig())
        self.config: FingerPointingConfig

    def candidate_names(self, features: GestureFeatures) -> list[str]:
        direction = index_pointing_direction(features, self.config.direction_ratio, self.config.min_vector_length)
        return [f"FINGER_POINT_{direction}"] if direction is not None else []

    def confidence(self, features: GestureFeatures, gesture: str) -> float:
        vector = _vector(features, idx.INDEX_MCP, idx.INDEX_TIP)
        return clamp_confidence(vector.magnitude() / max(self.config.min_vector_length, 1e-6))

    def _event(self, gesture: str, features: GestureFeatures, held_ms: float) -> GestureEvent:
        vector = _vector(features, idx.INDEX_MCP, idx.INDEX_TIP)
        direction = gesture.removeprefix("FINGER_POINT_")
        return GestureEvent(
            gesture=gesture,
            phase=GesturePhase.COMPLETED,
            confidence=self.confidence(features, gesture),
            timestamp=features.timestamp,
            position=features.position_payload(),
            velocity=features.velocity_payload(),
            metadata={
                "held_ms": held_ms,
                "extended_fingers": list(extended_fingers(features)),
                "finger_count": finger_count(features),
                "handedness": features.landmarks.handedness,
                "source": self.metadata_source,
                "direction": direction,
                "index_vector": vector.to_dict(),
            },
        )


def candidate_pose_names(features: GestureFeatures, config: StaticPoseConfig | None = None) -> list[str]:
    config = config or StaticPoseConfig()
    extended = set(extended_fingers(features))
    count = len(extended)
    candidates: list[str] = []

    if count == 5:
        candidates.append("OPEN_PALM")
    if count == 0:
        candidates.append("FIST")
    if is_point_index(features):
        candidates.append("POINT_INDEX")
    if is_two_finger_point(features):
        candidates.append("TWO_FINGER_POINT")
        if _distance(features, idx.INDEX_TIP, idx.MIDDLE_TIP) >= config.spread_distance_threshold:
            candidates.append("PEACE_SIGN")
    if is_ok_sign(features, config):
        candidates.append("OK_SIGN")
    if is_l_shape(features, config):
        candidates.append("L_SHAPE")
    if is_rock_sign(features):
        candidates.append("ROCK_SIGN")
    if is_hang_loose(features):
        candidates.append("HANG_LOOSE")
    if is_thumb_only(features):
        thumb_direction = thumb_vertical_direction(features, config.direction_ratio)
        if thumb_direction is not None:
            candidates.append(thumb_direction)
    return candidates


def pose_confidence(features: GestureFeatures, gesture: str, config: StaticPoseConfig) -> float:
    if gesture == "OK_SIGN":
        return clamp_confidence(1.0 - (features.pinch_distance / max(config.ok_distance_threshold, 1e-6)))
    if gesture.startswith("FINGER_COUNT_"):
        return 0.85
    if gesture.startswith("FLAT_HAND_"):
        return 0.8
    return 0.9


def extended_fingers(features: GestureFeatures) -> tuple[str, ...]:
    return tuple(finger for finger in FINGER_ORDER if not features.finger_curl.get(finger, True))


def finger_count(features: GestureFeatures) -> int:
    return len(extended_fingers(features))


def is_pinching(features: GestureFeatures, threshold: float = 0.12) -> bool:
    return features.pinch_distance <= threshold


def is_fist(features: GestureFeatures) -> bool:
    return finger_count(features) == 0


def is_point_index(features: GestureFeatures) -> bool:
    extended = set(extended_fingers(features))
    return "index" in extended and not {"middle", "ring", "pinky"} & extended


def is_two_finger_point(features: GestureFeatures) -> bool:
    extended = set(extended_fingers(features))
    return {"index", "middle"}.issubset(extended) and not {"ring", "pinky"} & extended


def is_ok_sign(features: GestureFeatures, config: StaticPoseConfig | None = None) -> bool:
    config = config or StaticPoseConfig()
    extended = set(extended_fingers(features))
    return (
        is_pinching(features, config.ok_distance_threshold)
        and {"middle", "ring", "pinky"}.issubset(extended)
    )


def is_thumb_only(features: GestureFeatures) -> bool:
    return set(extended_fingers(features)) == {"thumb"}


def is_l_shape(features: GestureFeatures, config: StaticPoseConfig | None = None) -> bool:
    config = config or StaticPoseConfig()
    extended = set(extended_fingers(features))
    if not {"thumb", "index"}.issubset(extended) or {"middle", "ring", "pinky"} & extended:
        return False
    angle = angle_between(
        _vector(features, idx.THUMB_MCP, idx.THUMB_TIP),
        _vector(features, idx.INDEX_MCP, idx.INDEX_TIP),
    )
    return config.l_shape_min_angle_degrees <= math.degrees(angle) <= config.l_shape_max_angle_degrees


def is_rock_sign(features: GestureFeatures) -> bool:
    extended = set(extended_fingers(features))
    return {"index", "pinky"}.issubset(extended) and not {"middle", "ring"} & extended


def is_hang_loose(features: GestureFeatures) -> bool:
    extended = set(extended_fingers(features))
    return {"thumb", "pinky"}.issubset(extended) and not {"index", "middle", "ring"} & extended


def thumb_vertical_direction(features: GestureFeatures, direction_ratio: float = 1.25) -> str | None:
    vector = _vector(features, idx.THUMB_MCP, idx.THUMB_TIP)
    if abs(vector.y) < abs(vector.x) * direction_ratio:
        return None
    return "THUMBS_UP" if vector.y < 0 else "THUMBS_DOWN"


def flat_hand_direction(features: GestureFeatures, direction_ratio: float = 1.25) -> str | None:
    vector = _vector(features, idx.WRIST, idx.MIDDLE_TIP)
    if abs(vector.x) >= abs(vector.y) * direction_ratio:
        return "FLAT_HAND_RIGHT" if vector.x > 0 else "FLAT_HAND_LEFT"
    if abs(vector.y) >= abs(vector.x) * direction_ratio:
        return "FLAT_HAND_DOWN" if vector.y > 0 else "FLAT_HAND_UP"
    return None


def index_pointing_direction(
    features: GestureFeatures,
    direction_ratio: float = 1.25,
    min_vector_length: float = 0.05,
) -> str | None:
    if not is_point_index(features):
        return None
    vector = _vector(features, idx.INDEX_MCP, idx.INDEX_TIP)
    if vector.magnitude() < min_vector_length:
        return None
    return classify_direction(vector, direction_ratio)


def hand_orientation_angle(features: GestureFeatures) -> float:
    vector = _vector(features, idx.WRIST, idx.MIDDLE_MCP)
    return math.atan2(vector.y, vector.x)


def angle_between(first: Point3D, second: Point3D) -> float:
    first_mag = max(first.magnitude(), 1e-6)
    second_mag = max(second.magnitude(), 1e-6)
    dot = first.x * second.x + first.y * second.y + first.z * second.z
    return math.acos(max(-1.0, min(1.0, dot / (first_mag * second_mag))))


def classify_direction(displacement: Point3D, direction_ratio: float = 1.35) -> str | None:
    abs_x = abs(displacement.x)
    abs_y = abs(displacement.y)
    if abs_x < 1e-9 and abs_y < 1e-9:
        return None
    if abs_x >= abs_y * direction_ratio:
        return "RIGHT" if displacement.x > 0 else "LEFT"
    if abs_y >= abs_x * direction_ratio:
        return "DOWN" if displacement.y > 0 else "UP"
    vertical = "DOWN" if displacement.y > 0 else "UP"
    horizontal = "RIGHT" if displacement.x > 0 else "LEFT"
    return f"{vertical}_{horizontal}"


def _vector(features: GestureFeatures, start_index: int, end_index: int) -> Point3D:
    start = features.landmarks.point(start_index)
    end = features.landmarks.point(end_index)
    return end.subtract(start)


def _distance(features: GestureFeatures, first_index: int, second_index: int) -> float:
    return features.landmarks.point(first_index).distance_to(features.landmarks.point(second_index))
