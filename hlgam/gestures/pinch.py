from __future__ import annotations

from dataclasses import dataclass

from hlgam.gestures.base import clamp_confidence
from hlgam.gestures.fsm import FSMState, FiniteStateMachine
from hlgam.models import GestureEvent, GestureFeatures, GesturePhase


@dataclass(frozen=True)
class PinchConfig:
    enabled: bool = True
    distance_threshold: float = 0.12
    release_threshold: float = 0.16
    hold_ms: int = 120


class PinchRecognizer:
    name = "pinch"

    def __init__(self, config: PinchConfig | None = None) -> None:
        self.config = config or PinchConfig()
        self._fsm = FiniteStateMachine()

    def process(self, features: GestureFeatures) -> list[GestureEvent]:
        if not self.config.enabled:
            return []
        events: list[GestureEvent] = []
        timestamp = features.timestamp
        is_pinching = features.pinch_distance <= self.config.distance_threshold
        is_released = features.pinch_distance >= self.config.release_threshold

        if self._fsm.state == FSMState.IDLE and is_pinching:
            self._fsm.transition(FSMState.CANDIDATE, timestamp)

        if self._fsm.state == FSMState.CANDIDATE:
            if not is_pinching:
                self._fsm.reset()
            elif self._fsm.elapsed(timestamp) * 1000.0 >= self.config.hold_ms:
                self._fsm.transition(FSMState.ACTIVE, timestamp)
                events.append(self._event("PINCH_START", GesturePhase.STARTED, features))

        if self._fsm.state == FSMState.ACTIVE:
            if is_released:
                events.append(self._event("PINCH_END", GesturePhase.ENDED, features))
                self._fsm.reset()
            elif not events:
                events.append(self._event("PINCH_HOLD", GesturePhase.UPDATED, features))

        return events

    def reset(self) -> None:
        self._fsm.reset()

    def _event(self, gesture: str, phase: GesturePhase, features: GestureFeatures) -> GestureEvent:
        confidence = 1.0 - (features.pinch_distance / max(self.config.release_threshold, 1e-6))
        return GestureEvent(
            gesture=gesture,
            phase=phase,
            confidence=clamp_confidence(confidence),
            timestamp=features.timestamp,
            position=features.position_payload(),
            velocity=features.velocity_payload(),
            metadata={"pinch_distance": features.pinch_distance},
        )
