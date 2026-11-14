from __future__ import annotations

from dataclasses import dataclass

from hlgam.gestures.base import clamp_confidence
from hlgam.gestures.drag import DragConfig, DragRecognizer
from hlgam.gestures.pinch import PinchConfig, PinchRecognizer
from hlgam.gestures.pose import is_pinching
from hlgam.models import GestureEvent, GestureFeatures, GesturePhase


@dataclass(frozen=True)
class PinchDragConfig:
    enabled: bool = True
    distance_threshold: float = 0.12
    release_threshold: float = 0.16
    hold_ms: int = 120
    drag_pinch_threshold: float = 0.12
    drag_release_threshold: float = 0.16
    drag_activation_hold_ms: int = 150
    drag_min_move: float = 0.002
    pinch_tap_distance_threshold: float = 0.12
    pinch_tap_release_threshold: float = 0.16
    pinch_tap_min_duration_ms: int = 20
    pinch_tap_max_duration_ms: int = 180


class PinchDragRecognizer:
    name = "pinch_drag"

    def __init__(self, config: PinchDragConfig | None = None) -> None:
        self.config = config or PinchDragConfig()
        self._pinch = PinchRecognizer(
            PinchConfig(
                enabled=True,
                distance_threshold=self.config.distance_threshold,
                release_threshold=self.config.release_threshold,
                hold_ms=self.config.hold_ms,
            )
        )
        self._drag = DragRecognizer(
            DragConfig(
                enabled=True,
                pinch_threshold=self.config.drag_pinch_threshold,
                release_threshold=self.config.drag_release_threshold,
                activation_hold_ms=self.config.drag_activation_hold_ms,
                min_move=self.config.drag_min_move,
            )
        )
        self._pinch_tap_started_at: float | None = None

    def process(self, features: GestureFeatures) -> list[GestureEvent]:
        if not self.config.enabled:
            return []

        events: list[GestureEvent] = []
        events.extend(self._pinch.process(features))
        events.extend(self._drag.process(features))
        events.extend(self._detect_pinch_tap(features))
        return events

    def reset(self) -> None:
        self._pinch.reset()
        self._drag.reset()
        self._pinch_tap_started_at = None

    def _detect_pinch_tap(self, features: GestureFeatures) -> list[GestureEvent]:
        events: list[GestureEvent] = []
        pinching = is_pinching(features, self.config.pinch_tap_distance_threshold)
        released = features.pinch_distance >= self.config.pinch_tap_release_threshold
        if pinching and self._pinch_tap_started_at is None:
            self._pinch_tap_started_at = features.timestamp
            return events
        if self._pinch_tap_started_at is None:
            return events

        duration_ms = (features.timestamp - self._pinch_tap_started_at) * 1000.0
        if released:
            if self.config.pinch_tap_min_duration_ms <= duration_ms <= self.config.pinch_tap_max_duration_ms:
                events.append(
                    GestureEvent(
                        gesture="PINCH_TAP",
                        phase=GesturePhase.COMPLETED,
                        confidence=clamp_confidence(
                            1.0 - (duration_ms / max(self.config.pinch_tap_max_duration_ms, 1))
                        ),
                        timestamp=features.timestamp,
                        position=features.position_payload(),
                        velocity=features.velocity_payload(),
                        metadata={"duration_ms": duration_ms},
                    )
                )
            self._pinch_tap_started_at = None
        elif duration_ms > self.config.pinch_tap_max_duration_ms:
            self._pinch_tap_started_at = None
        return events
