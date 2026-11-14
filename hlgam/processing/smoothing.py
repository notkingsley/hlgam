from __future__ import annotations

from dataclasses import dataclass

from hlgam.models import HandLandmarks, Point3D


@dataclass(frozen=True)
class SmoothingSettings:
    alpha: float = 0.65


class ExponentialLandmarkSmoother:
    def __init__(self, settings: SmoothingSettings | None = None) -> None:
        self.settings = settings or SmoothingSettings()
        if not 0.0 < self.settings.alpha <= 1.0:
            raise ValueError("alpha must be greater than 0 and less than or equal to 1")
        self._previous: dict[str, tuple[Point3D, ...]] = {}

    def smooth(self, hand: HandLandmarks) -> HandLandmarks:
        key = hand.handedness or "default"
        previous = self._previous.get(key)
        if previous is None:
            self._previous[key] = hand.landmarks
            return hand
        alpha = self.settings.alpha
        smoothed = tuple(
            Point3D(
                x=(alpha * current.x) + ((1.0 - alpha) * old.x),
                y=(alpha * current.y) + ((1.0 - alpha) * old.y),
                z=(alpha * current.z) + ((1.0 - alpha) * old.z),
            )
            for current, old in zip(hand.landmarks, previous, strict=True)
        )
        self._previous[key] = smoothed
        return hand.with_landmarks(smoothed)

    def reset(self, handedness: str | None = None) -> None:
        if handedness is None:
            self._previous.clear()
            return
        self._previous.pop(handedness, None)
