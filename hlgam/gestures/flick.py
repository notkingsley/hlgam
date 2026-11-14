from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from hlgam.gestures.base import clamp_confidence
from hlgam.gestures.pose import classify_direction
from hlgam.models import GestureEvent, GestureFeatures, GesturePhase


@dataclass(frozen=True)
class FlickConfig:
    enabled: bool = True
    history_size: int = 16
    max_history_ms: int = 1500
    cooldown_ms: int = 250
    direction_ratio: float = 1.35
    min_velocity: float = 1.2
    min_distance: float = 0.04
    max_duration_ms: int = 180


class FlickRecognizer:
    name = "flick"

    def __init__(self, config: FlickConfig | None = None) -> None:
        self.config = config or FlickConfig()
        self._history: deque[GestureFeatures] = deque(maxlen=max(2, self.config.history_size))
        self._cooldown_until = 0.0

    def process(self, features: GestureFeatures) -> list[GestureEvent]:
        if not self.config.enabled:
            return []

        self._history.append(features)
        if features.timestamp < self._cooldown_until:
            return []

        window = self._recent(self.config.max_duration_ms)
        if len(window) < 2 or features.speed < self.config.min_velocity:
            return []
        displacement = features.palm_center.subtract(window[0].palm_center)
        if displacement.magnitude() < self.config.min_distance:
            return []
        direction = classify_direction(displacement, self.config.direction_ratio)
        if direction is None:
            return []

        self._cooldown_until = features.timestamp + (self.config.cooldown_ms / 1000.0)
        return [
            GestureEvent(
                gesture=f"FLICK_{direction}",
                phase=GesturePhase.COMPLETED,
                confidence=clamp_confidence(features.speed / self.config.min_velocity),
                timestamp=features.timestamp,
                position=features.position_payload(),
                velocity=features.velocity_payload(),
                metadata={"displacement": displacement.to_dict(), "speed": features.speed},
            )
        ]

    def reset(self) -> None:
        self._history.clear()
        self._cooldown_until = 0.0

    def _recent(self, max_duration_ms: int) -> list[GestureFeatures]:
        if not self._history:
            return []
        current_time = self._history[-1].timestamp
        max_age = min(max_duration_ms, self.config.max_history_ms) / 1000.0
        return [item for item in self._history if current_time - item.timestamp <= max_age]
