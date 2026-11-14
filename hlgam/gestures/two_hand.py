from __future__ import annotations

import math
from dataclasses import dataclass

from hlgam.gestures.base import clamp_confidence
from hlgam.gestures.pose import classify_direction, is_l_shape
from hlgam.models import GestureEvent, GestureFeatures, GesturePhase, Point3D


@dataclass(frozen=True)
class TwoHandGestureConfig:
    enabled: bool = True
    cooldown_ms: int = 300
    zoom_min_delta: float = 0.10
    rotate_min_degrees: float = 35.0
    pan_min_distance: float = 0.10
    clap_distance_threshold: float = 0.14
    clap_min_closing_delta: float = 0.08
    frame_min_width: float = 0.30


class TwoHandGestureRecognizer:
    name = "two_hand"

    def __init__(self, config: TwoHandGestureConfig | None = None) -> None:
        self.config = config or TwoHandGestureConfig()
        self._reference: _PairReference | None = None
        self._cooldown_until = 0.0

    def process_many(self, features_list: list[GestureFeatures]) -> list[GestureEvent]:
        if not self.config.enabled or len(features_list) < 2:
            self.reset()
            return []

        pair = _ordered_pair(features_list)
        timestamp = max(pair.left.timestamp, pair.right.timestamp)
        metrics = _pair_metrics(pair)
        if self._reference is None:
            self._reference = _PairReference.from_metrics(metrics)
            return self._detect_static_pair(pair, metrics)

        events = self._detect_static_pair(pair, metrics)
        if timestamp < self._cooldown_until:
            return events

        reference = self._reference
        for event in (
            self._detect_clap(pair, metrics, reference),
            self._detect_two_hand_zoom(pair, metrics, reference),
            self._detect_two_hand_rotate(pair, metrics, reference),
            self._detect_two_hand_pan(pair, metrics, reference),
        ):
            if event is not None:
                self._reference = _PairReference.from_metrics(metrics)
                self._cooldown_until = timestamp + (self.config.cooldown_ms / 1000.0)
                events.append(event)
                break
        return events

    def reset(self) -> None:
        self._reference = None
        self._cooldown_until = 0.0

    def _detect_static_pair(self, pair: _HandPair, metrics: _PairMetrics) -> list[GestureEvent]:
        if metrics.distance < self.config.frame_min_width:
            return []
        if not is_l_shape(pair.left) or not is_l_shape(pair.right):
            return []
        return [
            _event(
                "FRAME_RECTANGLE",
                pair,
                confidence=metrics.distance / self.config.frame_min_width,
                metadata={"hand_distance": metrics.distance},
            )
        ]

    def _detect_clap(
        self,
        pair: _HandPair,
        metrics: _PairMetrics,
        reference: _PairReference,
    ) -> GestureEvent | None:
        closing_delta = reference.distance - metrics.distance
        if metrics.distance > self.config.clap_distance_threshold:
            return None
        if closing_delta < self.config.clap_min_closing_delta:
            return None
        return _event(
            "CLAP",
            pair,
            confidence=closing_delta / self.config.clap_min_closing_delta,
            metadata={"hand_distance": metrics.distance, "closing_delta": closing_delta},
        )

    def _detect_two_hand_zoom(
        self,
        pair: _HandPair,
        metrics: _PairMetrics,
        reference: _PairReference,
    ) -> GestureEvent | None:
        delta = metrics.distance - reference.distance
        if abs(delta) < self.config.zoom_min_delta:
            return None
        gesture = "TWO_HAND_ZOOM_IN" if delta > 0 else "TWO_HAND_ZOOM_OUT"
        return _event(
            gesture,
            pair,
            confidence=abs(delta) / self.config.zoom_min_delta,
            metadata={"hand_distance_delta": delta, "hand_distance": metrics.distance},
        )

    def _detect_two_hand_rotate(
        self,
        pair: _HandPair,
        metrics: _PairMetrics,
        reference: _PairReference,
    ) -> GestureEvent | None:
        rotation = _angle_delta(reference.angle, metrics.angle)
        min_rotation = math.radians(self.config.rotate_min_degrees)
        if abs(rotation) < min_rotation:
            return None
        gesture = "TWO_HAND_ROTATE_CLOCKWISE" if rotation > 0 else "TWO_HAND_ROTATE_COUNTERCLOCKWISE"
        return _event(
            gesture,
            pair,
            confidence=abs(rotation) / min_rotation,
            metadata={"rotation_degrees": math.degrees(rotation)},
        )

    def _detect_two_hand_pan(
        self,
        pair: _HandPair,
        metrics: _PairMetrics,
        reference: _PairReference,
    ) -> GestureEvent | None:
        displacement = metrics.center.subtract(reference.center)
        if displacement.magnitude() < self.config.pan_min_distance:
            return None
        direction = classify_direction(displacement)
        return _event(
            "TWO_HAND_PAN",
            pair,
            confidence=displacement.magnitude() / self.config.pan_min_distance,
            metadata={"direction": direction, "displacement": displacement.to_dict()},
        )


@dataclass(frozen=True)
class _HandPair:
    left: GestureFeatures
    right: GestureFeatures


@dataclass(frozen=True)
class _PairMetrics:
    center: Point3D
    distance: float
    angle: float


@dataclass(frozen=True)
class _PairReference:
    center: Point3D
    distance: float
    angle: float

    @classmethod
    def from_metrics(cls, metrics: _PairMetrics) -> _PairReference:
        return cls(center=metrics.center, distance=metrics.distance, angle=metrics.angle)


def _ordered_pair(features_list: list[GestureFeatures]) -> _HandPair:
    ordered = sorted(features_list[:2], key=lambda features: features.palm_center.x)
    return _HandPair(left=ordered[0], right=ordered[1])


def _pair_metrics(pair: _HandPair) -> _PairMetrics:
    left = pair.left.palm_center
    right = pair.right.palm_center
    center = Point3D((left.x + right.x) / 2.0, (left.y + right.y) / 2.0, (left.z + right.z) / 2.0)
    vector = right.subtract(left)
    return _PairMetrics(center=center, distance=left.distance_to(right), angle=math.atan2(vector.y, vector.x))


def _event(
    gesture: str,
    pair: _HandPair,
    *,
    confidence: float,
    metadata: dict[str, object] | None = None,
) -> GestureEvent:
    metrics = _pair_metrics(pair)
    return GestureEvent(
        gesture=gesture,
        phase=GesturePhase.COMPLETED,
        confidence=clamp_confidence(confidence),
        timestamp=max(pair.left.timestamp, pair.right.timestamp),
        position=metrics.center.to_dict(),
        velocity=None,
        metadata=metadata or {},
    )


def _angle_delta(start: float, end: float) -> float:
    delta = end - start
    while delta <= -math.pi:
        delta += math.tau
    while delta > math.pi:
        delta -= math.tau
    return delta
