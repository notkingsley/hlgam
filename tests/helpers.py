from __future__ import annotations

from hlgam import HandLandmarks, Point3D


HAND_SCALE = 0.10


def synthetic_hand(
    timestamp: float,
    normalized_pinch_distance: float = 0.08,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> HandLandmarks:
    pinch_distance = normalized_pinch_distance * HAND_SCALE
    points = [Point3D(offset_x, offset_y, 0.0) for _ in range(21)]
    points[0] = Point3D(offset_x, offset_y, 0.0)
    points[1] = Point3D(offset_x + 0.01, offset_y - 0.02, 0.0)
    points[2] = Point3D(offset_x + 0.015, offset_y - 0.04, 0.0)
    points[3] = Point3D(offset_x + 0.010, offset_y - 0.08, 0.0)
    points[4] = Point3D(offset_x, offset_y - 0.12, 0.0)
    points[5] = Point3D(offset_x + 0.08, offset_y - 0.04, 0.0)
    points[6] = Point3D(offset_x + 0.05, offset_y - 0.10, 0.0)
    points[7] = Point3D(offset_x + 0.04, offset_y - 0.13, 0.0)
    points[8] = Point3D(offset_x + pinch_distance, offset_y - 0.12, 0.0)
    points[9] = Point3D(offset_x, offset_y - HAND_SCALE, 0.0)
    points[10] = Point3D(offset_x, offset_y - 0.16, 0.0)
    points[11] = Point3D(offset_x, offset_y - 0.20, 0.0)
    points[12] = Point3D(offset_x, offset_y - 0.24, 0.0)
    points[13] = Point3D(offset_x - 0.07, offset_y - 0.04, 0.0)
    points[14] = Point3D(offset_x - 0.08, offset_y - 0.11, 0.0)
    points[15] = Point3D(offset_x - 0.08, offset_y - 0.16, 0.0)
    points[16] = Point3D(offset_x - 0.08, offset_y - 0.20, 0.0)
    points[17] = Point3D(offset_x - 0.10, offset_y, 0.0)
    points[18] = Point3D(offset_x - 0.12, offset_y - 0.07, 0.0)
    points[19] = Point3D(offset_x - 0.13, offset_y - 0.11, 0.0)
    points[20] = Point3D(offset_x - 0.14, offset_y - 0.15, 0.0)
    return HandLandmarks(tuple(points), timestamp=timestamp, handedness="right")
