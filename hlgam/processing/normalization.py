from __future__ import annotations

from dataclasses import dataclass

from hlgam import landmark_indices as idx
from hlgam.models import HandLandmarks


@dataclass(frozen=True)
class NormalizationSettings:
    min_scale: float = 1e-6


class HandNormalizer:
    def __init__(self, settings: NormalizationSettings | None = None) -> None:
        self.settings = settings or NormalizationSettings()

    def normalize(self, hand: HandLandmarks) -> HandLandmarks:
        wrist = hand.point(idx.WRIST)
        scale = self.scale_for(hand)
        normalized = tuple(point.subtract(wrist).scale(1.0 / scale) for point in hand.landmarks)
        return hand.with_landmarks(normalized)

    def scale_for(self, hand: HandLandmarks) -> float:
        wrist = hand.point(idx.WRIST)
        middle_mcp = hand.point(idx.MIDDLE_MCP)
        scale = wrist.distance_to(middle_mcp)
        if scale > self.settings.min_scale:
            return scale
        xs = [point.x for point in hand.landmarks]
        ys = [point.y for point in hand.landmarks]
        bbox_scale = max(max(xs) - min(xs), max(ys) - min(ys))
        return max(bbox_scale, self.settings.min_scale)
