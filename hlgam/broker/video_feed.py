from __future__ import annotations

import base64
import binascii
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from hlgam.models import Frame, GestureFeatures, HandLandmarks, Point3D
from hlgam.processing.feature_extractor import FeatureExtractor
from hlgam.processing.normalization import HandNormalizer
from hlgam.processing.smoothing import ExponentialLandmarkSmoother
from hlgam.tracking.hand_tracker import HandTrackerSettings, MediaPipeHandTracker


INTERNAL_VIDEO_SOURCE_ID = "internal_camera"
JPEG_BASE64_ENCODING = "jpeg_base64"
VALID_SOURCE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


class VideoFeedError(ValueError):
    pass


@dataclass(frozen=True)
class VideoOutputOptions:
    include_frame: bool = True
    include_landmarks: bool = True
    include_features: bool = True
    jpeg_quality: int = 80

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> VideoOutputOptions:
        return cls(
            include_frame=bool_value(payload.get("include_frame"), True, "include_frame"),
            include_landmarks=bool_value(payload.get("include_landmarks"), True, "include_landmarks"),
            include_features=bool_value(payload.get("include_features"), True, "include_features"),
            jpeg_quality=int_range_value(payload.get("jpeg_quality"), 80, 1, 100, "jpeg_quality"),
        )


@dataclass
class VideoSourceProcessor:
    normalizer: HandNormalizer = field(default_factory=HandNormalizer)
    smoother: ExponentialLandmarkSmoother = field(default_factory=ExponentialLandmarkSmoother)
    extractors: dict[str, FeatureExtractor] = field(default_factory=dict)

    def extract_features(self, hands: Sequence[HandLandmarks]) -> list[GestureFeatures]:
        if not hands:
            self.reset()
            return []

        features_list: list[GestureFeatures] = []
        for index, hand in enumerate(hands):
            smoothed = self.smoother.smooth(hand)
            normalized = self.normalizer.normalize(smoothed)
            extractor = self._extractor_for(smoothed, index)
            features_list.append(extractor.extract(normalized, raw_landmarks=smoothed))
        return features_list

    def reset(self) -> None:
        self.smoother.reset()
        self.extractors.clear()

    def _extractor_for(self, hand: HandLandmarks, index: int) -> FeatureExtractor:
        key = hand.handedness or f"hand_{index}"
        if key not in self.extractors:
            self.extractors[key] = FeatureExtractor()
        return self.extractors[key]


@dataclass
class VideoSourceState:
    source_id: str
    label: str
    kind: str
    owner: object | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_frame_at: float | None = None
    frame_index: int = 0
    width: int | None = None
    height: int | None = None
    processor: VideoSourceProcessor = field(default_factory=VideoSourceProcessor)
    tracker_settings: HandTrackerSettings | None = None
    tracker: MediaPipeHandTracker | None = None

    def info_payload(self) -> dict[str, Any]:
        return {
            "id": self.source_id,
            "label": self.label,
            "kind": self.kind,
            "created_at": self.created_at,
            "last_frame_at": self.last_frame_at,
            "frame_index": self.frame_index,
            "width": self.width,
            "height": self.height,
            "metadata": self.metadata,
        }

    def update_frame_metadata(self, frame: Frame | None) -> None:
        self.last_frame_at = time.time()
        if frame is None:
            self.frame_index += 1
            return
        self.frame_index = frame.index or self.frame_index + 1
        shape = getattr(frame.image, "shape", None)
        if shape is not None and len(shape) >= 2:
            self.height = int(shape[0])
            self.width = int(shape[1])

    def track(self, frame: Frame) -> list[HandLandmarks]:
        if self.tracker is None:
            self.tracker = MediaPipeHandTracker(self.tracker_settings)
        return self.tracker.track(frame)

    def close(self) -> None:
        if self.tracker is not None:
            self.tracker.close()
        self.tracker = None


def validate_source_id(source_id: Any) -> str:
    if not isinstance(source_id, str):
        raise VideoFeedError("source_id must be a string")
    source_id = source_id.strip()
    if not VALID_SOURCE_ID.fullmatch(source_id):
        raise VideoFeedError(
            "source_id must be 1-64 characters using letters, numbers, underscore, dash, dot, or colon"
        )
    return source_id


def bool_value(value: Any, default: bool, name: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise VideoFeedError(f"{name} must be a boolean")
    return value


def int_range_value(value: Any, default: int, minimum: int, maximum: int, name: str) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise VideoFeedError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise VideoFeedError(f"{name} must be between {minimum} and {maximum}")
    return value


def string_value(value: Any, default: str, name: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise VideoFeedError(f"{name} must be a non-empty string")
    return value.strip()


def metadata_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise VideoFeedError("metadata must be an object")
    return dict(value)


def decode_frame_from_message(payload: dict[str, Any]) -> Frame | None:
    frame_payload = payload.get("frame")
    if frame_payload is None and "data" not in payload:
        return None
    if frame_payload is None:
        frame_payload = payload
    if not isinstance(frame_payload, dict):
        raise VideoFeedError("frame must be an object")

    data = frame_payload.get("data")
    if data is None:
        return None
    if not isinstance(data, str) or not data:
        raise VideoFeedError("frame.data must be a non-empty base64 string")

    encoding = string_value(
        frame_payload.get("encoding", payload.get("encoding")),
        JPEG_BASE64_ENCODING,
        "frame.encoding",
    )
    if encoding not in {JPEG_BASE64_ENCODING, "jpg_base64", "png_base64"}:
        raise VideoFeedError("frame.encoding must be jpeg_base64, jpg_base64, or png_base64")

    timestamp = float_value(
        frame_payload.get("timestamp", payload.get("timestamp")),
        time.perf_counter(),
        "frame.timestamp",
    )
    index = int_value(frame_payload.get("index", payload.get("frame_index")), 0, "frame.index")
    return Frame(image=decode_image_data(data), timestamp=timestamp, index=index)


def decode_image_data(data: str) -> Any:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise VideoFeedError("raw video input requires opencv-python and numpy") from exc

    if "," in data and data.lstrip().startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise VideoFeedError("frame.data is not valid base64") from exc
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise VideoFeedError("frame.data could not be decoded as an image")
    return image


def encode_frame_payload(frame: Frame, jpeg_quality: int = 80) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise VideoFeedError("video output requires opencv-python") from exc

    ok, encoded = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise VideoFeedError("frame could not be encoded as JPEG")

    shape = getattr(frame.image, "shape", None)
    height = int(shape[0]) if shape is not None and len(shape) >= 2 else None
    width = int(shape[1]) if shape is not None and len(shape) >= 2 else None
    return {
        "encoding": JPEG_BASE64_ENCODING,
        "mime_type": "image/jpeg",
        "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
        "width": width,
        "height": height,
        "index": frame.index,
        "timestamp": frame.timestamp,
    }


def parse_hands_from_message(payload: dict[str, Any], frame: Frame | None = None) -> list[HandLandmarks] | None:
    hands_payload = payload.get("hands")
    if hands_payload is None and "landmarks" in payload:
        hands_payload = [payload]
    if hands_payload is None:
        return None
    if not isinstance(hands_payload, list):
        raise VideoFeedError("hands must be an array")

    timestamp = float_value(
        payload.get("timestamp"),
        frame.timestamp if frame is not None else time.perf_counter(),
        "timestamp",
    )
    image_size = image_size_for_frame(frame)
    return [parse_hand_payload(item, timestamp, image_size) for item in hands_payload]


def parse_hand_payload(
    payload: Any,
    default_timestamp: float,
    default_image_size: tuple[int, int] | None,
) -> HandLandmarks:
    if isinstance(payload, list):
        landmarks_payload = payload
        handedness = None
        score = None
        timestamp = default_timestamp
        image_size = default_image_size
    elif isinstance(payload, dict):
        landmarks_payload = payload.get("landmarks")
        handedness = payload.get("handedness")
        score = payload.get("score")
        timestamp = float_value(payload.get("timestamp"), default_timestamp, "hand.timestamp")
        image_size = parse_image_size(payload.get("image_size"), default_image_size)
    else:
        raise VideoFeedError("each hand must be an object or landmark array")

    if not isinstance(landmarks_payload, list):
        raise VideoFeedError("hand.landmarks must be an array")
    points = tuple(parse_point(point) for point in landmarks_payload)
    if len(points) != 21:
        raise VideoFeedError(f"hand.landmarks must contain 21 points, received {len(points)}")
    if handedness is not None and not isinstance(handedness, str):
        raise VideoFeedError("hand.handedness must be a string")
    if score is not None and not isinstance(score, (int, float)):
        raise VideoFeedError("hand.score must be a number")
    return HandLandmarks(
        landmarks=points,
        timestamp=timestamp,
        handedness=handedness.lower() if isinstance(handedness, str) else None,
        score=float(score) if score is not None else None,
        image_size=image_size,
    )


def parse_point(payload: Any) -> Point3D:
    if isinstance(payload, dict):
        try:
            return Point3D(float(payload["x"]), float(payload["y"]), float(payload.get("z", 0.0)))
        except (KeyError, TypeError, ValueError) as exc:
            raise VideoFeedError("landmark points must include numeric x and y values") from exc
    if isinstance(payload, list | tuple) and len(payload) >= 2:
        try:
            return Point3D(float(payload[0]), float(payload[1]), float(payload[2]) if len(payload) > 2 else 0.0)
        except (TypeError, ValueError) as exc:
            raise VideoFeedError("landmark point arrays must contain numeric values") from exc
    raise VideoFeedError("landmark points must be objects or arrays")


def image_size_for_frame(frame: Frame | None) -> tuple[int, int] | None:
    if frame is None:
        return None
    shape = getattr(frame.image, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    return int(shape[1]), int(shape[0])


def parse_image_size(value: Any, default: tuple[int, int] | None) -> tuple[int, int] | None:
    if value is None:
        return default
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise VideoFeedError("hand.image_size must be [width, height]")
    width = int_value(value[0], 0, "hand.image_size[0]")
    height = int_value(value[1], 0, "hand.image_size[1]")
    return width, height


def hand_payload(hand: HandLandmarks) -> dict[str, Any]:
    return {
        "landmarks": [point.to_dict() for point in hand.landmarks],
        "timestamp": hand.timestamp,
        "handedness": hand.handedness,
        "score": hand.score,
        "image_size": hand.image_size,
    }


def features_payload(features: GestureFeatures) -> dict[str, Any]:
    return {
        "timestamp": features.timestamp,
        "scale": features.scale,
        "palm_center": features.palm_center.to_dict(),
        "pinch_distance": features.pinch_distance,
        "pinch_delta": features.pinch_delta,
        "velocity": features.velocity.to_dict(),
        "speed": features.speed,
        "direction": features.direction.to_dict(),
        "finger_curl": features.finger_curl,
        "bounding_box": features.bounding_box,
        "handedness": features.raw_landmarks.handedness,
    }


def float_value(value: Any, default: float, name: str) -> float:
    if value is None:
        return default
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise VideoFeedError(f"{name} must be a number")
    return float(value)


def int_value(value: Any, default: int, name: str) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise VideoFeedError(f"{name} must be an integer")
    return value
