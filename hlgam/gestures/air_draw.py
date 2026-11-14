from __future__ import annotations

from dataclasses import dataclass

from hlgam import landmark_indices as idx
from hlgam.gestures.base import clamp_confidence
from hlgam.gestures.pose import is_point_index
from hlgam.models import GestureEvent, GestureFeatures, GesturePhase, Point3D


@dataclass(frozen=True)
class AirDrawConfig:
    enabled: bool = True
    min_distance: float = 0.01


class AirDrawRecognizer:
    name = "air_draw"

    def __init__(self, config: AirDrawConfig | None = None) -> None:
        self.config = config or AirDrawConfig()
        self._active = False
        self._last_point: Point3D | None = None

    def process(self, features: GestureFeatures) -> list[GestureEvent]:
        if not self.config.enabled:
            return []

        events: list[GestureEvent] = []
        drawing = is_point_index(features)
        point = _path_point(features)
        if drawing and not self._active:
            self._active = True
            self._last_point = point
            events.append(self._event("AIR_DRAW_START", features, confidence=0.9, metadata={"point": point.to_dict()}))
            return events
        if drawing and self._active:
            previous = self._last_point or point
            distance = point.distance_to(previous)
            if distance >= self.config.min_distance:
                self._last_point = point
                events.append(
                    self._event(
                        "AIR_DRAW_MOVE",
                        features,
                        confidence=distance / self.config.min_distance,
                        metadata={"point": point.to_dict(), "distance": distance},
                    )
                )
            return events
        if not drawing and self._active:
            self._active = False
            self._last_point = None
            events.append(self._event("AIR_DRAW_END", features, confidence=0.9, metadata={"point": point.to_dict()}))
        return events

    def reset(self) -> None:
        self._active = False
        self._last_point = None

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
