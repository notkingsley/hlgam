from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GesturePhase(str, Enum):
    STARTED = "started"
    UPDATED = "updated"
    COMPLETED = "completed"
    ENDED = "ended"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float = 0.0

    def distance_to(self, other: Point3D) -> float:
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )

    def add(self, other: Point3D) -> Point3D:
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def subtract(self, other: Point3D) -> Point3D:
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, factor: float) -> Point3D:
        return Point3D(self.x * factor, self.y * factor, self.z * factor)

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> Point3D:
        magnitude = self.magnitude()
        if magnitude == 0:
            return Point3D(0.0, 0.0, 0.0)
        return self.scale(1.0 / magnitude)

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass(frozen=True)
class Frame:
    image: Any
    timestamp: float
    index: int = 0


@dataclass(frozen=True)
class HandLandmarks:
    landmarks: tuple[Point3D, ...]
    timestamp: float
    handedness: str | None = None
    score: float | None = None
    image_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        points = tuple(self.landmarks)
        if len(points) != 21:
            raise ValueError(f"expected 21 hand landmarks, received {len(points)}")
        object.__setattr__(self, "landmarks", points)

    def point(self, index: int) -> Point3D:
        return self.landmarks[index]

    def with_landmarks(self, landmarks: tuple[Point3D, ...]) -> HandLandmarks:
        return HandLandmarks(
            landmarks=landmarks,
            timestamp=self.timestamp,
            handedness=self.handedness,
            score=self.score,
            image_size=self.image_size,
        )


@dataclass(frozen=True)
class GestureFeatures:
    landmarks: HandLandmarks
    raw_landmarks: HandLandmarks
    timestamp: float
    scale: float
    palm_center: Point3D
    pinch_distance: float
    pinch_delta: float
    velocity: Point3D
    speed: float
    direction: Point3D
    finger_curl: dict[str, bool]
    bounding_box: tuple[float, float, float, float]

    def position_payload(self) -> dict[str, float]:
        return self.palm_center.to_dict()

    def velocity_payload(self) -> dict[str, float]:
        return self.velocity.to_dict()


@dataclass(frozen=True)
class GestureEvent:
    gesture: str
    phase: GesturePhase | str
    confidence: float
    timestamp: float
    position: dict[str, float] | None = None
    velocity: dict[str, float] | None = None
    command: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        phase = self.phase.value if isinstance(self.phase, GesturePhase) else self.phase
        return {
            "gesture": self.gesture,
            "phase": phase,
            "confidence": max(0.0, min(1.0, self.confidence)),
            "timestamp": self.timestamp,
            "position": self.position,
            "velocity": self.velocity,
            "command": self.command,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))
