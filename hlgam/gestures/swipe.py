from __future__ import annotations

from dataclasses import dataclass
from hlgam.gestures.base import clamp_confidence
from hlgam.gestures.fsm import FSMState, FiniteStateMachine
from hlgam.gestures.pose import classify_direction
from hlgam.models import GestureEvent, GestureFeatures, GesturePhase, Point3D


@dataclass(frozen=True)
class SwipeMotionConfig:
    enabled: bool = True
    min_velocity: float = 0.65
    min_distance: float = 0.12
    diagonal_swipe_min_distance: float = 0.16
    diagonal_swipe_axis_balance_ratio: float = 2.25
    max_duration_ms: int = 500
    cooldown_ms: int = 300
    direction_ratio: float = 1.35


class SwipeMotionRecognizer:
    name = "swipe_motion"

    def __init__(self, config: SwipeMotionConfig | None = None) -> None:
        self.config = config or SwipeMotionConfig()
        self._fsm = FiniteStateMachine()
        self._start_position: Point3D | None = None
        self._cooldown_until = 0.0

    def process(self, features: GestureFeatures) -> list[GestureEvent]:
        if not self.config.enabled or features.timestamp < self._cooldown_until:
            return []

        timestamp = features.timestamp
        if self._fsm.state == FSMState.IDLE and features.speed >= self.config.min_velocity:
            self._fsm.transition(FSMState.CANDIDATE, timestamp)
            self._start_position = features.palm_center
            return []

        if self._fsm.state != FSMState.CANDIDATE:
            return []

        if self._fsm.elapsed(timestamp) * 1000.0 > self.config.max_duration_ms:
            self.reset()
            return []

        if self._start_position is None:
            self.reset()
            return []

        displacement = features.palm_center.subtract(self._start_position)
        distance = displacement.magnitude()
        classified = self._classify_direction(displacement)
        if classified is None:
            self.reset()
            return []
        gesture, required_distance = classified
        if distance < required_distance:
            return []

        confidence = min(distance / required_distance, features.speed / self.config.min_velocity) / 2.0 + 0.5
        event = GestureEvent(
            gesture=gesture,
            phase=GesturePhase.COMPLETED,
            confidence=clamp_confidence(confidence),
            timestamp=timestamp,
            position=features.position_payload(),
            velocity=features.velocity_payload(),
            metadata={"distance": distance, "displacement": displacement.to_dict()},
        )
        self._cooldown_until = timestamp + (self.config.cooldown_ms / 1000.0)
        self.reset(keep_cooldown=True)
        return [event]

    def reset(self, keep_cooldown: bool = False) -> None:
        self._fsm.reset()
        self._start_position = None
        if not keep_cooldown:
            self._cooldown_until = 0.0

    def _classify_direction(self, displacement: Point3D) -> tuple[str, float] | None:
        abs_x = abs(displacement.x)
        abs_y = abs(displacement.y)
        direction = classify_direction(displacement, self.config.direction_ratio)
        if direction is None:
            return None
        if "_" not in direction:
            return f"SWIPE_{direction}", self.config.min_distance
        if abs_x <= 0.0 or abs_y <= 0.0:
            return None
        balance = max(abs_x, abs_y) / max(min(abs_x, abs_y), 1e-6)
        if balance > self.config.diagonal_swipe_axis_balance_ratio:
            return None
        return f"DIAGONAL_SWIPE_{direction}", self.config.diagonal_swipe_min_distance


SwipeConfig = SwipeMotionConfig
SwipeRecognizer = SwipeMotionRecognizer
