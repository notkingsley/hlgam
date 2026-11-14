from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from hlgam.gestures.base import clamp_confidence
from hlgam.gestures.pose import classify_direction, is_two_finger_point
from hlgam.models import GestureEvent, GestureFeatures, GesturePhase


@dataclass(frozen=True)
class ScrollConfig:
    enabled: bool = True
    history_size: int = 32
    max_history_ms: int = 1500
    cooldown_ms: int = 250
    direction_ratio: float = 1.35
    min_distance: float = 0.08
    max_duration_ms: int = 900


class ScrollRecognizer:
    name = "scroll"

    def __init__(self, config: ScrollConfig | None = None) -> None:
        self.config = config or ScrollConfig()
        self._history: deque[GestureFeatures] = deque(maxlen=max(2, self.config.history_size))
        self._cooldown_until = 0.0

    def process(self, features: GestureFeatures) -> list[GestureEvent]:
        if not self.config.enabled:
            return []

        self._history.append(features)
        if features.timestamp < self._cooldown_until:
            return []

        event = self._detect_scroll(features)
        if event is None:
            return []

        self._cooldown_until = features.timestamp + (self.config.cooldown_ms / 1000.0)
        return [event]

    def reset(self) -> None:
        self._history.clear()
        self._cooldown_until = 0.0

    def _detect_scroll(self, features: GestureFeatures) -> GestureEvent | None:
        if not is_two_finger_point(features):
            return None
        window = [item for item in self._recent(self.config.max_duration_ms) if is_two_finger_point(item)]
        if len(window) < 2:
            return None

        displacement = features.palm_center.subtract(window[0].palm_center)
        direction = classify_direction(displacement, self.config.direction_ratio)
        if direction not in {"UP", "DOWN", "LEFT", "RIGHT"}:
            return None

        axis_distance = abs(displacement.x) if direction in {"LEFT", "RIGHT"} else abs(displacement.y)
        if axis_distance < self.config.min_distance:
            return None

        return GestureEvent(
            gesture=f"SCROLL_{direction}",
            phase=GesturePhase.COMPLETED,
            confidence=clamp_confidence(axis_distance / self.config.min_distance),
            timestamp=features.timestamp,
            position=features.position_payload(),
            velocity=features.velocity_payload(),
            metadata={"direction": direction, "displacement": displacement.to_dict()},
        )

    def _recent(self, max_duration_ms: int) -> list[GestureFeatures]:
        if not self._history:
            return []
        current_time = self._history[-1].timestamp
        max_age = min(max_duration_ms, self.config.max_history_ms) / 1000.0
        return [item for item in self._history if current_time - item.timestamp <= max_age]
