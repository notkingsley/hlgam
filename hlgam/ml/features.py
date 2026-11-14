from __future__ import annotations

from hlgam.ml.schemas import LandmarkFeatureSpec
from hlgam.models import GestureFeatures, HandLandmarks, Point3D


def landmark_matrix(features: GestureFeatures, spec: LandmarkFeatureSpec) -> list[list[float]]:
    hand = feature_hand(features, spec.coordinate_space)
    return [[float(getattr(point, axis)) for axis in spec.landmark_axes] for point in hand.landmarks]


def global_feature_vector(features: GestureFeatures, spec: LandmarkFeatureSpec) -> list[float]:
    return [global_feature_value(features, name) for name in spec.global_feature_names]


def feature_hand(features: GestureFeatures, coordinate_space: str) -> HandLandmarks:
    if coordinate_space == "normalized":
        return features.landmarks
    if coordinate_space == "raw":
        return features.raw_landmarks
    raise ValueError(f"unsupported coordinate space: {coordinate_space}")


def global_feature_value(features: GestureFeatures, name: str) -> float:
    if name == "feature_scale":
        return float(features.scale)
    if name == "feature_pinch_distance":
        return float(features.pinch_distance)
    if name == "feature_pinch_delta":
        return float(features.pinch_delta)
    if name == "feature_speed":
        return float(features.speed)
    if name.startswith("feature_palm_center_"):
        return point_axis_value(features.palm_center, name.removeprefix("feature_palm_center_"))
    if name.startswith("feature_velocity_"):
        return point_axis_value(features.velocity, name.removeprefix("feature_velocity_"))
    if name.startswith("feature_direction_"):
        return point_axis_value(features.direction, name.removeprefix("feature_direction_"))
    if name.startswith("feature_bbox_"):
        return bbox_value(features.bounding_box, name.removeprefix("feature_bbox_"))
    if name.startswith("feature_finger_curl_"):
        finger = name.removeprefix("feature_finger_curl_")
        return 1.0 if features.finger_curl.get(finger, True) else 0.0
    raise ValueError(f"unsupported static model global feature: {name}")


def point_axis_value(point: Point3D, axis: str) -> float:
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"unsupported point axis: {axis}")
    return float(getattr(point, axis))


def bbox_value(bounding_box: tuple[float, float, float, float], name: str) -> float:
    mapping = {
        "min_x": bounding_box[0],
        "min_y": bounding_box[1],
        "max_x": bounding_box[2],
        "max_y": bounding_box[3],
    }
    if name not in mapping:
        raise ValueError(f"unsupported bounding-box feature: {name}")
    return float(mapping[name])
