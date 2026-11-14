from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from hlgam import landmark_indices as idx
from hlgam.gestures.base import clamp_confidence
from hlgam.gestures.pose import hand_orientation_angle, is_point_index
from hlgam.models import GestureEvent, GestureFeatures, GesturePhase, Point3D


@dataclass(frozen=True)
class RotationConfig:
    enabled: bool = True
    history_size: int = 32
    max_history_ms: int = 1500
    cooldown_ms: int = 250
    circle_min_radius: float = 0.05
    circle_min_arc_degrees: float = 300.0
    circle_max_duration_ms: int = 1400
    circle_closure_ratio: float = 1.5
    rotate_min_degrees: float = 70.0
    rotate_max_duration_ms: int = 1000


class RotationRecognizer:
    name = "rotation"

    def __init__(self, config: RotationConfig | None = None) -> None:
        self.config = config or RotationConfig()
        self._history: deque[GestureFeatures] = deque(maxlen=max(2, self.config.history_size))
        self._cooldown_until = 0.0

    def process(self, features: GestureFeatures) -> list[GestureEvent]:
        if not self.config.enabled:
            return []

        self._history.append(features)
        if features.timestamp < self._cooldown_until:
            return []

        for detector in (self._detect_circle, self._detect_rotate):
            event = detector(features)
            if event is not None:
                self._cooldown_until = features.timestamp + (self.config.cooldown_ms / 1000.0)
                return [event]
        return []

    def reset(self) -> None:
        self._history.clear()
        self._cooldown_until = 0.0

    def _detect_circle(self, features: GestureFeatures) -> GestureEvent | None:
        window = self._recent(self.config.circle_max_duration_ms)
        if len(window) < 6:
            return None
        points = [_path_point(item) for item in window]
        center = Point3D(
            sum(point.x for point in points) / len(points),
            sum(point.y for point in points) / len(points),
            sum(point.z for point in points) / len(points),
        )
        radii = [point.distance_to(center) for point in points]
        radius = sum(radii) / len(radii)
        if radius < self.config.circle_min_radius:
            return None
        if points[0].distance_to(points[-1]) > radius * self.config.circle_closure_ratio:
            return None

        angles = [math.atan2(point.y - center.y, point.x - center.x) for point in points]
        arc = _unwrapped_angle_delta(angles)
        min_arc = math.radians(self.config.circle_min_arc_degrees)
        if abs(arc) < min_arc:
            return None
        gesture = "CIRCLE_CLOCKWISE" if arc > 0 else "CIRCLE_COUNTERCLOCKWISE"
        return self._event(
            gesture,
            features,
            confidence=abs(arc) / min_arc,
            metadata={"arc_degrees": math.degrees(arc), "radius": radius},
        )

    def _detect_rotate(self, features: GestureFeatures) -> GestureEvent | None:
        window = self._recent(self.config.rotate_max_duration_ms)
        if len(window) < 2:
            return None
        angles = [hand_orientation_angle(item) for item in window]
        rotation = _unwrapped_angle_delta(angles)
        min_rotation = math.radians(self.config.rotate_min_degrees)
        if abs(rotation) < min_rotation:
            return None
        gesture = "ROTATE_CLOCKWISE" if rotation > 0 else "ROTATE_COUNTERCLOCKWISE"
        return self._event(
            gesture,
            features,
            confidence=abs(rotation) / min_rotation,
            metadata={"rotation_degrees": math.degrees(rotation)},
        )

    def _recent(self, max_duration_ms: int) -> list[GestureFeatures]:
        if not self._history:
            return []
        current_time = self._history[-1].timestamp
        max_age = min(max_duration_ms, self.config.max_history_ms) / 1000.0
        return [item for item in self._history if current_time - item.timestamp <= max_age]

    def _event(
        self,
        gesture: str,
        features: GestureFeatures,
        *,
        confidence: float,
        metadata: dict[str, object] | None = None,
    ) -> GestureEvent:
        return GestureEvent(
            gesture=gesture,
            phase=GesturePhase.COMPLETED,
            confidence=clamp_confidence(confidence),
            timestamp=features.timestamp,
            position=features.position_payload(),
            velocity=features.velocity_payload(),
            metadata=metadata or {},
        )


def _path_point(features: GestureFeatures) -> Point3D:
    if is_point_index(features):
        return features.raw_landmarks.point(idx.INDEX_TIP)
    return features.palm_center


def _unwrapped_angle_delta(angles: list[float]) -> float:
    total = 0.0
    previous = angles[0]
    for angle in angles[1:]:
        delta = angle - previous
        while delta <= -math.pi:
            delta += math.tau
        while delta > math.pi:
            delta -= math.tau
        total += delta
        previous = angle
    return total
