from __future__ import annotations

from dataclasses import dataclass

from hlgam.models import Frame, HandLandmarks, Point3D


@dataclass(frozen=True)
class HandTrackerSettings:
    max_num_hands: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    model_complexity: int = 1


class MediaPipeHandTracker:
    def __init__(self, settings: HandTrackerSettings | None = None) -> None:
        self.settings = settings or HandTrackerSettings()
        self._cv2 = None
        self._mp_hands = None
        self._hands = None

    def open(self) -> None:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe tracking requires mediapipe and opencv-python optional dependencies") from exc

        self._cv2 = cv2
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.settings.max_num_hands,
            model_complexity=self.settings.model_complexity,
            min_detection_confidence=self.settings.min_detection_confidence,
            min_tracking_confidence=self.settings.min_tracking_confidence,
        )

    def track(self, frame: Frame) -> list[HandLandmarks]:
        if self._hands is None:
            self.open()
        assert self._cv2 is not None
        assert self._hands is not None

        rgb_image = self._cv2.cvtColor(frame.image, self._cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb_image)
        if not result.multi_hand_landmarks:
            return []

        image_height, image_width = frame.image.shape[:2]
        hands: list[HandLandmarks] = []
        handedness_items = result.multi_handedness or []
        for index, hand_landmarks in enumerate(result.multi_hand_landmarks):
            handedness = None
            score = None
            if index < len(handedness_items):
                classification = handedness_items[index].classification[0]
                handedness = classification.label.lower()
                score = classification.score
            hands.append(
                HandLandmarks(
                    landmarks=tuple(Point3D(point.x, point.y, point.z) for point in hand_landmarks.landmark),
                    timestamp=frame.timestamp,
                    handedness=handedness,
                    score=score,
                    image_size=(image_width, image_height),
                )
            )
        return hands

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
        self._hands = None

    def __enter__(self) -> MediaPipeHandTracker:
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
