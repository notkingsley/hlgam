from __future__ import annotations

from hlgam import GestureFramework, HandLandmarks, Point3D


HAND_SCALE = 0.10


def synthetic_hand(timestamp: float, normalized_pinch_distance: float, offset_x: float = 0.0) -> HandLandmarks:
    pinch_distance = normalized_pinch_distance * HAND_SCALE
    points = [Point3D(offset_x, 0.0, 0.0) for _ in range(21)]
    points[0] = Point3D(offset_x, 0.0, 0.0)
    points[5] = Point3D(offset_x + 0.08, -0.04, 0.0)
    points[9] = Point3D(offset_x, -0.10, 0.0)
    points[13] = Point3D(offset_x - 0.07, -0.04, 0.0)
    points[17] = Point3D(offset_x - 0.10, 0.00, 0.0)
    points[3] = Point3D(offset_x + 0.02, -0.06, 0.0)
    points[4] = Point3D(offset_x, -0.12, 0.0)
    points[6] = Point3D(offset_x + 0.03, -0.13, 0.0)
    points[8] = Point3D(offset_x + pinch_distance, -0.12, 0.0)
    return HandLandmarks(tuple(points), timestamp=timestamp, handedness="right")


def main() -> None:
    framework = GestureFramework()
    framework.on_any(lambda event: print(event.to_json()))
    for timestamp, distance in [(0.0, 0.08), (0.13, 0.08), (0.20, 0.08), (0.30, 0.24)]:
        framework.process_landmarks(synthetic_hand(timestamp, distance))


if __name__ == "__main__":
    main()
