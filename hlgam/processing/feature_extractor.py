from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from hlgam import landmark_indices as idx
from hlgam.models import GestureFeatures, HandLandmarks, Point3D
from hlgam.processing.normalization import HandNormalizer


@dataclass(frozen=True)
class FeatureExtractorSettings:
    history_size: int = 8
    min_dt: float = 1e-6


class FeatureExtractor:
    def __init__(self, settings: FeatureExtractorSettings | None = None) -> None:
        self.settings = settings or FeatureExtractorSettings()
        if self.settings.history_size < 2:
            raise ValueError("history_size must be at least 2")
        self._history: deque[GestureFeatures] = deque(maxlen=self.settings.history_size)
        self._scale_helper = HandNormalizer()

    def extract(self, landmarks: HandLandmarks, raw_landmarks: HandLandmarks | None = None) -> GestureFeatures:
        raw = raw_landmarks or landmarks
        palm_center = self._palm_center(raw)
        pinch_distance = landmarks.point(idx.THUMB_TIP).distance_to(landmarks.point(idx.INDEX_TIP))
        bounding_box = self._bounding_box(raw)
        scale = self._scale_helper.scale_for(raw)

        velocity = Point3D(0.0, 0.0, 0.0)
        pinch_delta = 0.0
        if self._history:
            previous = self._history[-1]
            dt = max(raw.timestamp - previous.timestamp, self.settings.min_dt)
            velocity = palm_center.subtract(previous.palm_center).scale(1.0 / dt)
            pinch_delta = pinch_distance - previous.pinch_distance

        speed = velocity.magnitude()
        direction = velocity.normalized()
        features = GestureFeatures(
            landmarks=landmarks,
            raw_landmarks=raw,
            timestamp=raw.timestamp,
            scale=scale,
            palm_center=palm_center,
            pinch_distance=pinch_distance,
            pinch_delta=pinch_delta,
            velocity=velocity,
            speed=speed,
            direction=direction,
            finger_curl=self._finger_curl(landmarks),
            bounding_box=bounding_box,
        )
        self._history.append(features)
        return features

    def reset(self) -> None:
        self._history.clear()

    def _palm_center(self, hand: HandLandmarks) -> Point3D:
        points = [hand.point(index) for index in idx.PALM_INDICES]
        factor = 1.0 / len(points)
        return Point3D(
            x=sum(point.x for point in points) * factor,
            y=sum(point.y for point in points) * factor,
            z=sum(point.z for point in points) * factor,
        )

    def _bounding_box(self, hand: HandLandmarks) -> tuple[float, float, float, float]:
        xs = [point.x for point in hand.landmarks]
        ys = [point.y for point in hand.landmarks]
        return min(xs), min(ys), max(xs), max(ys)

    def _finger_curl(self, hand: HandLandmarks) -> dict[str, bool]:
        wrist = hand.point(idx.WRIST)
        curls: dict[str, bool] = {}
        for finger, tip_index in idx.FINGER_TIPS.items():
            pip_index = idx.FINGER_PIPS[finger]
            tip_distance = hand.point(tip_index).distance_to(wrist)
            pip_distance = hand.point(pip_index).distance_to(wrist)
            curls[finger] = tip_distance < pip_distance
        return curls
