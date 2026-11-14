from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


HANDEDNESS_TO_ID = {
    "unknown": 0,
    "left": 1,
    "right": 2,
}

DEFAULT_STATIC_GLOBAL_FEATURES = (
    "feature_scale",
    "feature_palm_center_x",
    "feature_palm_center_y",
    "feature_palm_center_z",
    "feature_pinch_distance",
    "feature_pinch_delta",
    "feature_velocity_x",
    "feature_velocity_y",
    "feature_velocity_z",
    "feature_speed",
    "feature_direction_x",
    "feature_direction_y",
    "feature_direction_z",
    "feature_bbox_min_x",
    "feature_bbox_min_y",
    "feature_bbox_max_x",
    "feature_bbox_max_y",
    "feature_finger_curl_thumb",
    "feature_finger_curl_index",
    "feature_finger_curl_middle",
    "feature_finger_curl_ring",
    "feature_finger_curl_pinky",
)

DEFAULT_MOTION_GLOBAL_FEATURES = DEFAULT_STATIC_GLOBAL_FEATURES


@dataclass(frozen=True)
class LabelSchema:
    schema_version: int
    model_family: str
    labels: tuple[str, ...]
    multi_label: bool = True
    compatible_overlaps: tuple[tuple[str, ...], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        labels = tuple(self.labels)
        if not labels:
            raise ValueError("label schema must contain at least one label")
        duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
        if duplicate_labels:
            raise ValueError(f"duplicate labels in schema: {', '.join(duplicate_labels)}")
        object.__setattr__(self, "labels", labels)
        overlaps = tuple(tuple(overlap) for overlap in self.compatible_overlaps)
        object.__setattr__(self, "compatible_overlaps", overlaps)

    @classmethod
    def from_path(cls, path: Path | str) -> LabelSchema:
        with Path(path).open("r", encoding="utf-8") as schema_file:
            payload = json.load(schema_file)
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> LabelSchema:
        labels = payload.get("labels")
        if not isinstance(labels, Sequence) or isinstance(labels, str):
            raise ValueError("label schema requires a labels array")
        overlaps = payload.get("compatible_overlaps", ())
        if not isinstance(overlaps, Sequence) or isinstance(overlaps, str):
            raise ValueError("compatible_overlaps must be an array when supplied")
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            model_family=str(payload.get("model_family", "")),
            labels=tuple(str(label) for label in labels),
            multi_label=bool(payload.get("multi_label", True)),
            compatible_overlaps=tuple(tuple(str(label) for label in overlap) for overlap in overlaps),
        )

    @property
    def label_to_index(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.labels)}

    def encode(self, labels: Iterable[str]) -> list[float]:
        vector = [0.0 for _ in self.labels]
        label_to_index = self.label_to_index
        unknown = sorted({label for label in labels if label not in label_to_index})
        if unknown:
            raise ValueError(f"labels not present in schema: {', '.join(unknown)}")
        for label in labels:
            vector[label_to_index[label]] = 1.0
        return vector

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_family": self.model_family,
            "multi_label": self.multi_label,
            "labels": list(self.labels),
            "compatible_overlaps": [list(overlap) for overlap in self.compatible_overlaps],
        }

    def write(self, path: Path | str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as schema_file:
            json.dump(self.to_dict(), schema_file, indent=2)
            schema_file.write("\n")


@dataclass(frozen=True)
class LandmarkFeatureSpec:
    coordinate_space: str = "normalized"
    num_landmarks: int = 21
    landmark_axes: tuple[str, ...] = ("x", "y", "z")
    global_feature_names: tuple[str, ...] = DEFAULT_STATIC_GLOBAL_FEATURES
    handedness_classes: tuple[str, ...] = ("unknown", "left", "right")

    def __post_init__(self) -> None:
        if self.coordinate_space not in {"normalized", "raw"}:
            raise ValueError("coordinate_space must be normalized or raw")
        if self.num_landmarks <= 0:
            raise ValueError("num_landmarks must be greater than zero")
        if not self.landmark_axes:
            raise ValueError("landmark_axes must not be empty")
        object.__setattr__(self, "landmark_axes", tuple(self.landmark_axes))
        object.__setattr__(self, "global_feature_names", tuple(self.global_feature_names))
        object.__setattr__(self, "handedness_classes", tuple(self.handedness_classes))

    @property
    def per_landmark_dim(self) -> int:
        return len(self.landmark_axes)

    @property
    def global_feature_dim(self) -> int:
        return len(self.global_feature_names)

    def landmark_column(self, landmark_index: int, axis: str) -> str:
        return f"{self.coordinate_space}_landmark_{landmark_index:02d}_{axis}"

    def landmark_columns(self) -> tuple[str, ...]:
        return tuple(
            self.landmark_column(index, axis)
            for index in range(self.num_landmarks)
            for axis in self.landmark_axes
        )

    def handedness_id(self, handedness: str | None) -> int:
        normalized = (handedness or "unknown").strip().casefold()
        return HANDEDNESS_TO_ID.get(normalized, HANDEDNESS_TO_ID["unknown"])

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> LandmarkFeatureSpec:
        return cls(
            coordinate_space=str(payload.get("coordinate_space", "normalized")),
            num_landmarks=int(payload.get("num_landmarks", 21)),
            landmark_axes=tuple(str(axis) for axis in payload.get("landmark_axes", ("x", "y", "z"))),
            global_feature_names=tuple(
                str(name) for name in payload.get("global_feature_names", DEFAULT_STATIC_GLOBAL_FEATURES)
            ),
            handedness_classes=tuple(
                str(name) for name in payload.get("handedness_classes", ("unknown", "left", "right"))
            ),
        )


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    model_family: str
    architecture: str
    architecture_version: int
    model_file: str
    label_schema: dict[str, object]
    input_feature_schema: dict[str, object]
    thresholds_file: str | None = None
    metrics_summary: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def write(self, path: Path | str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as manifest_file:
            json.dump(self.to_dict(), manifest_file, indent=2)
            manifest_file.write("\n")
