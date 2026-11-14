from __future__ import annotations

import csv
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from hlgam.ml.schemas import DEFAULT_STATIC_GLOBAL_FEATURES, LandmarkFeatureSpec, LabelSchema


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATIC_LABEL_SCHEMA = PROJECT_ROOT / "training" / "label_schemas" / "static_pose_labels.json"
TRUTHY_VALUES = {"1", "true", "yes", "y"}
FALSY_VALUES = {"", "0", "false", "no", "n"}

POINT_DIRECTION_TO_ID = {
    "none": 0,
    "FINGER_POINT_LEFT": 1,
    "FINGER_POINT_RIGHT": 2,
    "FINGER_POINT_UP": 3,
    "FINGER_POINT_DOWN": 4,
    "FINGER_POINT_UP_LEFT": 5,
    "FINGER_POINT_UP_RIGHT": 6,
    "FINGER_POINT_DOWN_LEFT": 7,
    "FINGER_POINT_DOWN_RIGHT": 8,
}
FLAT_DIRECTION_TO_ID = {
    "none": 0,
    "FLAT_HAND_LEFT": 1,
    "FLAT_HAND_RIGHT": 2,
    "FLAT_HAND_UP": 3,
    "FLAT_HAND_DOWN": 4,
}


@dataclass(frozen=True)
class StaticPoseDatasetProfile:
    csv_path: Path
    total_rows: int
    label_counts: dict[str, int]
    primary_pose_counts: dict[str, int]
    handedness_counts: dict[str, int]
    missing_image_count: int
    missing_landmark_rows: int
    unlabeled_rows: int
    duplicate_capture_ids: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "csv_path": str(self.csv_path),
            "total_rows": self.total_rows,
            "label_counts": self.label_counts,
            "primary_pose_counts": self.primary_pose_counts,
            "handedness_counts": self.handedness_counts,
            "missing_image_count": self.missing_image_count,
            "missing_landmark_rows": self.missing_landmark_rows,
            "unlabeled_rows": self.unlabeled_rows,
            "duplicate_capture_ids": list(self.duplicate_capture_ids),
        }

    def write_json(self, path: Path | str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as profile_file:
            json.dump(self.to_dict(), profile_file, indent=2)
            profile_file.write("\n")


class StaticPoseDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        csv_path: Path | str,
        *,
        label_schema_path: Path | str = DEFAULT_STATIC_LABEL_SCHEMA,
        feature_spec: LandmarkFeatureSpec | None = None,
        strict: bool = True,
        limit_rows: int | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.label_schema = LabelSchema.from_path(label_schema_path)
        if self.label_schema.model_family and self.label_schema.model_family != "static_pose":
            raise ValueError("static pose dataset requires a static_pose label schema")
        self.feature_spec = feature_spec or LandmarkFeatureSpec(global_feature_names=DEFAULT_STATIC_GLOBAL_FEATURES)
        self.strict = strict
        self.rows = self._load_rows(limit_rows)
        if not self.rows:
            raise ValueError(f"static pose CSV contains no rows: {self.csv_path}")

    @property
    def num_labels(self) -> int:
        return len(self.label_schema.labels)

    @property
    def landmark_dim(self) -> int:
        return self.feature_spec.per_landmark_dim

    @property
    def global_feature_dim(self) -> int:
        return self.feature_spec.global_feature_dim

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        label_set = parse_label_set(row, self.label_schema, strict=self.strict)
        labels = torch.tensor(self.label_schema.encode(label_set), dtype=torch.float32)
        landmarks = torch.tensor(parse_landmarks(row, self.feature_spec), dtype=torch.float32)
        global_features = torch.tensor(parse_global_features(row, self.feature_spec), dtype=torch.float32)
        handedness = torch.tensor(self.feature_spec.handedness_id(row.get("target_handedness")), dtype=torch.long)
        landmark_mask = torch.ones(self.feature_spec.num_landmarks, dtype=torch.bool)

        return {
            "landmarks": landmarks,
            "global_features": global_features,
            "handedness": handedness,
            "landmark_mask": landmark_mask,
            "labels": labels,
            "aux": build_auxiliary_targets(label_set),
            "metadata": build_metadata(row),
        }

    def profile(self) -> StaticPoseDatasetProfile:
        return profile_rows(self.csv_path, self.rows, self.label_schema, self.feature_spec, strict=self.strict)

    def _load_rows(self, limit_rows: int | None) -> list[dict[str, str]]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"static pose CSV not found: {self.csv_path}")
        with self.csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise ValueError(f"static pose CSV has no header: {self.csv_path}")
            validate_header(reader.fieldnames, self.label_schema, self.feature_spec)
            rows = [row for row_index, row in enumerate(reader) if limit_rows is None or row_index < limit_rows]
        return rows


def validate_header(fieldnames: Sequence[str], label_schema: LabelSchema, feature_spec: LandmarkFeatureSpec) -> None:
    present = set(fieldnames)
    required = {
        "capture_id",
        "primary_pose",
        "target_handedness",
        "labels",
        "image_path",
        *feature_spec.landmark_columns(),
        *feature_spec.global_feature_names,
    }
    required.update(f"label_{label}" for label in label_schema.labels)
    missing = sorted(required - present)
    if missing:
        preview = ", ".join(missing[:12])
        suffix = "..." if len(missing) > 12 else ""
        raise ValueError(f"static pose CSV is missing required columns: {preview}{suffix}")


def parse_label_set(row: Mapping[str, str], label_schema: LabelSchema, *, strict: bool) -> set[str]:
    labels: set[str] = set()
    labels_value = (row.get("labels") or "").strip()
    if labels_value:
        labels.update(label.strip() for label in labels_value.split("|") if label.strip())

    for label in label_schema.labels:
        value = (row.get(f"label_{label}") or "").strip().casefold()
        if value in TRUTHY_VALUES:
            labels.add(label)
        elif value not in FALSY_VALUES and strict:
            raise ValueError(f"invalid binary value for label_{label}: {value!r}")

    unknown = sorted(labels - set(label_schema.labels))
    if unknown and strict:
        raise ValueError(f"row contains labels not present in schema: {', '.join(unknown)}")
    return labels - set(unknown)


def parse_landmarks(row: Mapping[str, str], feature_spec: LandmarkFeatureSpec) -> list[list[float]]:
    landmarks: list[list[float]] = []
    for landmark_index in range(feature_spec.num_landmarks):
        point = [
            parse_float(row, feature_spec.landmark_column(landmark_index, axis))
            for axis in feature_spec.landmark_axes
        ]
        landmarks.append(point)
    return landmarks


def parse_global_features(row: Mapping[str, str], feature_spec: LandmarkFeatureSpec) -> list[float]:
    return [parse_float(row, feature_name) for feature_name in feature_spec.global_feature_names]


def parse_float(row: Mapping[str, str], column: str) -> float:
    value = (row.get(column) or "").strip()
    if value == "":
        return 0.0
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"invalid numeric value for {column}: {value!r}") from exc


def build_auxiliary_targets(label_set: set[str]) -> dict[str, torch.Tensor]:
    return {
        "finger_count": torch.tensor(finger_count_target(label_set), dtype=torch.long),
        "point_direction": torch.tensor(direction_target(label_set, POINT_DIRECTION_TO_ID), dtype=torch.long),
        "flat_direction": torch.tensor(direction_target(label_set, FLAT_DIRECTION_TO_ID), dtype=torch.long),
    }


def finger_count_target(label_set: set[str]) -> int:
    for count in range(1, 6):
        if f"FINGER_COUNT_{count}" in label_set:
            return count
    return -100


def direction_target(label_set: set[str], mapping: Mapping[str, int]) -> int:
    for label, target_id in mapping.items():
        if label != "none" and label in label_set:
            return target_id
    return 0


def build_metadata(row: Mapping[str, str]) -> dict[str, str]:
    metadata_columns = (
        "capture_id",
        "captured_at_utc",
        "primary_pose",
        "target_handedness",
        "detected_handedness",
        "detected_handedness_score",
        "image_path",
        "hand_selection_note",
    )
    return {column: row.get(column, "") for column in metadata_columns}


def static_pose_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "landmarks": torch.stack([sample["landmarks"] for sample in batch]),
        "global_features": torch.stack([sample["global_features"] for sample in batch]),
        "handedness": torch.stack([sample["handedness"] for sample in batch]),
        "landmark_mask": torch.stack([sample["landmark_mask"] for sample in batch]),
        "labels": torch.stack([sample["labels"] for sample in batch]),
        "aux": {
            "finger_count": torch.stack([sample["aux"]["finger_count"] for sample in batch]),
            "point_direction": torch.stack([sample["aux"]["point_direction"] for sample in batch]),
            "flat_direction": torch.stack([sample["aux"]["flat_direction"] for sample in batch]),
        },
        "metadata": [sample["metadata"] for sample in batch],
    }


def split_dataset_indices(
    dataset_size: int,
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 1337,
) -> dict[str, list[int]]:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be greater than zero")
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must be non-negative and sum to less than one")
    indices = list(range(dataset_size))
    random.Random(seed).shuffle(indices)
    test_count = int(round(dataset_size * test_fraction))
    val_count = int(round(dataset_size * val_fraction))
    if dataset_size >= 3 and val_fraction > 0:
        val_count = max(1, val_count)
    if dataset_size >= 3 and test_fraction > 0:
        test_count = max(1, test_count)
    if val_count + test_count >= dataset_size:
        overflow = val_count + test_count - dataset_size + 1
        val_count = max(0, val_count - overflow)
    test_indices = indices[:test_count]
    val_indices = indices[test_count:test_count + val_count]
    train_indices = indices[test_count + val_count:]
    if not train_indices:
        raise ValueError("split configuration produced an empty training split")
    return {"train": train_indices, "validation": val_indices, "test": test_indices}


def build_static_pose_dataloaders(
    csv_path: Path | str,
    *,
    label_schema_path: Path | str = DEFAULT_STATIC_LABEL_SCHEMA,
    feature_spec: LandmarkFeatureSpec | None = None,
    batch_size: int = 64,
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 1337,
    num_workers: int = 0,
    limit_rows: int | None = None,
    pin_memory: bool = False,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
) -> tuple[dict[str, DataLoader[dict[str, Any]]], StaticPoseDataset, dict[str, list[int]]]:
    dataset = StaticPoseDataset(
        csv_path,
        label_schema_path=label_schema_path,
        feature_spec=feature_spec,
        limit_rows=limit_rows,
    )
    split_indices = split_dataset_indices(
        len(dataset),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    dataloaders: dict[str, DataLoader[dict[str, Any]]] = {}
    for split_name, indices in split_indices.items():
        if not indices:
            continue
        subset = Subset(dataset, indices)
        sampler = None
        shuffle = split_name == "train"
        if split_name == "train" and distributed_world_size > 1:
            sampler = DistributedSampler(
                subset,
                num_replicas=distributed_world_size,
                rank=distributed_rank,
                shuffle=True,
                seed=seed,
                drop_last=False,
            )
            shuffle = False
        dataloaders[split_name] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=static_pose_collate,
            pin_memory=pin_memory,
        )
    return dataloaders, dataset, split_indices


def compute_static_pos_weight(dataset: Dataset[dict[str, Any]]) -> torch.Tensor:
    labels = [dataset[index]["labels"] for index in range(len(dataset))]
    stacked = torch.stack(labels)
    positives = stacked.sum(dim=0)
    negatives = stacked.shape[0] - positives
    return torch.where(positives > 0, negatives / positives.clamp_min(1.0), torch.ones_like(positives))


def profile_static_pose_csv(
    csv_path: Path | str,
    *,
    label_schema_path: Path | str = DEFAULT_STATIC_LABEL_SCHEMA,
    feature_spec: LandmarkFeatureSpec | None = None,
    strict: bool = True,
) -> StaticPoseDatasetProfile:
    dataset = StaticPoseDataset(
        csv_path,
        label_schema_path=label_schema_path,
        feature_spec=feature_spec,
        strict=strict,
    )
    return dataset.profile()


def profile_rows(
    csv_path: Path,
    rows: Iterable[Mapping[str, str]],
    label_schema: LabelSchema,
    feature_spec: LandmarkFeatureSpec,
    *,
    strict: bool,
) -> StaticPoseDatasetProfile:
    label_counts = Counter({label: 0 for label in label_schema.labels})
    primary_pose_counts: Counter[str] = Counter()
    handedness_counts: Counter[str] = Counter()
    capture_ids: Counter[str] = Counter()
    missing_image_count = 0
    missing_landmark_rows = 0
    unlabeled_rows = 0
    total_rows = 0

    for row in rows:
        total_rows += 1
        label_set = parse_label_set(row, label_schema, strict=strict)
        if not label_set:
            unlabeled_rows += 1
        label_counts.update(label_set)
        primary_pose_counts.update([(row.get("primary_pose") or "unknown")])
        handedness_counts.update([(row.get("target_handedness") or "unknown").casefold()])
        capture_id = row.get("capture_id") or ""
        if capture_id:
            capture_ids.update([capture_id])
        if not (row.get("image_path") or "").strip():
            missing_image_count += 1
        if row_has_missing_landmark(row, feature_spec):
            missing_landmark_rows += 1

    duplicate_capture_ids = tuple(sorted(capture_id for capture_id, count in capture_ids.items() if count > 1))
    return StaticPoseDatasetProfile(
        csv_path=csv_path,
        total_rows=total_rows,
        label_counts=dict(label_counts),
        primary_pose_counts=dict(primary_pose_counts),
        handedness_counts=dict(handedness_counts),
        missing_image_count=missing_image_count,
        missing_landmark_rows=missing_landmark_rows,
        unlabeled_rows=unlabeled_rows,
        duplicate_capture_ids=duplicate_capture_ids,
    )


def row_has_missing_landmark(row: Mapping[str, str], feature_spec: LandmarkFeatureSpec) -> bool:
    return any((row.get(column) or "").strip() == "" for column in feature_spec.landmark_columns())
