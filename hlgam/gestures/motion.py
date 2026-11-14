from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from hlgam.gestures.base import clamp_confidence
from hlgam.gestures.pose import (
    classify_direction,
    is_fist,
)
from hlgam.models import GestureEvent, GestureFeatures, GesturePhase, Point3D


@dataclass(frozen=True)
class MotionGestureConfig:
    enabled: bool = True
    history_size: int = 32
    max_history_ms: int = 1500
    cooldown_ms: int = 250
    direction_ratio: float = 1.35
    wave_min_amplitude: float = 0.08
    wave_min_direction_changes: int = 3
    wave_max_duration_ms: int = 1200
    push_pull_min_scale_delta: float = 0.18
    push_pull_max_duration_ms: int = 900
    throw_min_distance: float = 0.12
    throw_min_velocity: float = 0.8
    throw_max_duration_ms: int = 900


class MotionGestureRecognizer:
    name = "motion"

    def __init__(self, config: MotionGestureConfig | None = None) -> None:
        self.config = config or MotionGestureConfig()
        self._history: deque[GestureFeatures] = deque(maxlen=max(2, self.config.history_size))
        self._cooldown_until = 0.0
        self._grab_started_at: float | None = None
        self._grab_start_position: Point3D | None = None
        self._grab_peak_velocity = Point3D(0.0, 0.0, 0.0)

    def process(self, features: GestureFeatures) -> list[GestureEvent]:
        if not self.config.enabled:
            return []

        self._history.append(features)
        events: list[GestureEvent] = []
        events.extend(self._detect_grab_and_throw(features))

        if features.timestamp < self._cooldown_until:
            return events

        for detector in (self._detect_wave, self._detect_push_pull):
            event = detector(features)
            if event is not None:
                self._cooldown_until = features.timestamp + (self.config.cooldown_ms / 1000.0)
                events.append(event)
                break
        return events

    def reset(self) -> None:
        self._history.clear()
        self._cooldown_until = 0.0
        self._grab_started_at = None
        self._grab_start_position = None
        self._grab_peak_velocity = Point3D(0.0, 0.0, 0.0)

    def _detect_wave(self, features: GestureFeatures) -> GestureEvent | None:
        window = self._recent(self.config.wave_max_duration_ms)
        if len(window) < 5:
            return None
        xs = [item.palm_center.x for item in window]
        amplitude = max(xs) - min(xs)
        if amplitude < self.config.wave_min_amplitude:
            return None

        direction_changes = 0
        previous_sign = 0
        for previous, current in zip(xs, xs[1:]):
            delta = current - previous
            if abs(delta) < self.config.wave_min_amplitude * 0.15:
                continue
            sign = 1 if delta > 0 else -1
            if previous_sign and sign != previous_sign:
                direction_changes += 1
            previous_sign = sign
        if direction_changes < self.config.wave_min_direction_changes:
            return None
        return self._event(
            "WAVE",
            features,
            confidence=(direction_changes + 1) / max(self.config.wave_min_direction_changes + 1, 1),
            metadata={"amplitude": amplitude, "direction_changes": direction_changes},
        )

    def _detect_push_pull(self, features: GestureFeatures) -> GestureEvent | None:
        window = self._recent(self.config.push_pull_max_duration_ms)
        if len(window) < 2:
            return None
        start_scale = max(window[0].scale, 1e-6)
        scale_delta = (features.scale - start_scale) / start_scale
        if abs(scale_delta) < self.config.push_pull_min_scale_delta:
            return None
        gesture = "PUSH" if scale_delta > 0 else "PULL"
        return self._event(
            gesture,
            features,
            confidence=abs(scale_delta) / self.config.push_pull_min_scale_delta,
            metadata={"scale_delta": scale_delta, "start_scale": start_scale, "current_scale": features.scale},
        )

    def _detect_grab_and_throw(self, features: GestureFeatures) -> list[GestureEvent]:
        events: list[GestureEvent] = []
        fist = is_fist(features)
        if fist and self._grab_started_at is None:
            self._grab_started_at = features.timestamp
            self._grab_start_position = features.palm_center
            self._grab_peak_velocity = features.velocity
            return events
        if fist:
            if features.speed > self._grab_peak_velocity.magnitude():
                self._grab_peak_velocity = features.velocity
            return events
        if self._grab_started_at is None or self._grab_start_position is None:
            return events

        duration_ms = (features.timestamp - self._grab_started_at) * 1000.0
        displacement = features.palm_center.subtract(self._grab_start_position)
        peak_speed = self._grab_peak_velocity.magnitude()
        if (
            duration_ms <= self.config.throw_max_duration_ms
            and displacement.magnitude() >= self.config.throw_min_distance
            and peak_speed >= self.config.throw_min_velocity
        ):
            direction = classify_direction(displacement, self.config.direction_ratio)
            events.append(
                self._event(
                    "GRAB_AND_THROW",
                    features,
                    confidence=min(
                        displacement.magnitude() / self.config.throw_min_distance,
                        peak_speed / self.config.throw_min_velocity,
                    ),
                    metadata={
                        "direction": direction,
                        "displacement": displacement.to_dict(),
                        "duration_ms": duration_ms,
                        "peak_velocity": self._grab_peak_velocity.to_dict(),
                    },
                )
            )
        self._grab_started_at = None
        self._grab_start_position = None
        self._grab_peak_velocity = Point3D(0.0, 0.0, 0.0)
        return events

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
