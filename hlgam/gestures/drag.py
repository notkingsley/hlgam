from __future__ import annotations

from dataclasses import dataclass
from hlgam.gestures.base import clamp_confidence
from hlgam.gestures.fsm import FSMState, FiniteStateMachine
from hlgam.models import GestureEvent, GestureFeatures, GesturePhase, Point3D


@dataclass(frozen=True)
class DragConfig:
    enabled: bool = True
    pinch_threshold: float = 0.12
    release_threshold: float = 0.16
    activation_hold_ms: int = 150
    min_move: float = 0.002


class DragRecognizer:
    name = "drag"

    def __init__(self, config: DragConfig | None = None) -> None:
        self.config = config or DragConfig()
        self._fsm = FiniteStateMachine()
        self._last_position: Point3D | None = None

    def process(self, features: GestureFeatures) -> list[GestureEvent]:
        if not self.config.enabled:
            return []
        events: list[GestureEvent] = []
        timestamp = features.timestamp
        is_pinching = features.pinch_distance <= self.config.pinch_threshold
        is_released = features.pinch_distance >= self.config.release_threshold

        if self._fsm.state == FSMState.IDLE and is_pinching:
            self._fsm.transition(FSMState.CANDIDATE, timestamp)
            self._last_position = features.palm_center

        if self._fsm.state == FSMState.CANDIDATE:
            if is_released:
                self.reset()
            elif self._fsm.elapsed(timestamp) * 1000.0 >= self.config.activation_hold_ms:
                self._fsm.transition(FSMState.ACTIVE, timestamp)
                self._last_position = features.palm_center
                events.append(self._event("DRAG_START", GesturePhase.STARTED, features, Point3D(0.0, 0.0, 0.0)))

        if self._fsm.state == FSMState.ACTIVE:
            if is_released:
                delta = self._delta(features.palm_center)
                events.append(self._event("DRAG_END", GesturePhase.ENDED, features, delta))
                self.reset()
            elif not events:
                delta = self._delta(features.palm_center)
                if delta.magnitude() >= self.config.min_move:
                    events.append(self._event("DRAG_MOVE", GesturePhase.UPDATED, features, delta))
                    self._last_position = features.palm_center

        return events

    def reset(self) -> None:
        self._fsm.reset()
        self._last_position = None

    def _delta(self, position: Point3D) -> Point3D:
        if self._last_position is None:
            return Point3D(0.0, 0.0, 0.0)
        return position.subtract(self._last_position)

    def _event(self, gesture: str, phase: GesturePhase, features: GestureFeatures, delta: Point3D) -> GestureEvent:
        confidence = 1.0 - (features.pinch_distance / max(self.config.release_threshold, 1e-6))
        return GestureEvent(
            gesture=gesture,
            phase=phase,
            confidence=clamp_confidence(confidence),
            timestamp=features.timestamp,
            position=features.position_payload(),
            velocity=features.velocity_payload(),
            metadata={"pinch_distance": features.pinch_distance, "delta": delta.to_dict()},
        )
