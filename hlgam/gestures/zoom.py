from __future__ import annotations

from dataclasses import dataclass
from hlgam.gestures.base import clamp_confidence
from hlgam.gestures.fsm import FSMState, FiniteStateMachine
from hlgam.models import GestureEvent, GestureFeatures, GesturePhase


@dataclass(frozen=True)
class ZoomConfig:
    enabled: bool = True
    min_reference_distance: float = 0.04
    min_delta: float = 0.08
    min_duration_ms: int = 120
    max_duration_ms: int = 700
    cooldown_ms: int = 250


class ZoomRecognizer:
    name = "zoom"

    def __init__(self, config: ZoomConfig | None = None) -> None:
        self.config = config or ZoomConfig()
        self._fsm = FiniteStateMachine()
        self._start_distance: float | None = None
        self._started_at = 0.0
        self._active_direction: int | None = None
        self._cooldown_until = 0.0

    def process(self, features: GestureFeatures) -> list[GestureEvent]:
        if not self.config.enabled or features.timestamp < self._cooldown_until:
            return []

        timestamp = features.timestamp
        if self._fsm.state == FSMState.IDLE and features.pinch_distance >= self.config.min_reference_distance:
            self._fsm.transition(FSMState.CANDIDATE, timestamp)
            self._start_distance = features.pinch_distance
            self._started_at = timestamp
            self._active_direction = None
            return []

        if self._fsm.state not in {FSMState.CANDIDATE, FSMState.ACTIVE}:
            return []

        if (timestamp - self._started_at) * 1000.0 > self.config.max_duration_ms:
            self.reset()
            return []

        if self._start_distance is None:
            self.reset()
            return []

        delta = features.pinch_distance - self._start_distance
        direction = self._direction_for(delta)
        if direction == 0:
            if self._fsm.state == FSMState.ACTIVE:
                self.reset()
            return []

        if self._fsm.state == FSMState.CANDIDATE:
            self._active_direction = direction
            self._fsm.transition(FSMState.ACTIVE, timestamp)
            return []

        if direction != self._active_direction:
            self.reset()
            return []

        duration_ms = self._fsm.elapsed(timestamp) * 1000.0
        if duration_ms < self.config.min_duration_ms:
            return []

        gesture = "ZOOM_IN" if direction > 0 else "ZOOM_OUT"
        event = GestureEvent(
            gesture=gesture,
            phase=GesturePhase.COMPLETED,
            confidence=clamp_confidence(abs(delta) / self.config.min_delta),
            timestamp=timestamp,
            position=features.position_payload(),
            velocity=features.velocity_payload(),
            metadata={
                "pinch_delta": delta,
                "pinch_distance": features.pinch_distance,
                "zoom_duration_ms": duration_ms,
            },
        )
        self._cooldown_until = timestamp + (self.config.cooldown_ms / 1000.0)
        self.reset(keep_cooldown=True)
        return [event]

    def reset(self, keep_cooldown: bool = False) -> None:
        self._fsm.reset()
        self._start_distance = None
        self._started_at = 0.0
        self._active_direction = None
        if not keep_cooldown:
            self._cooldown_until = 0.0

    def _direction_for(self, delta: float) -> int:
        if delta >= self.config.min_delta:
            return 1
        if delta <= -self.config.min_delta:
            return -1
        return 0
