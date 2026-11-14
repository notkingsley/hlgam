from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from hlgam.ml.schemas import DEFAULT_MOTION_GLOBAL_FEATURES, LandmarkFeatureSpec, LabelSchema


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOTION_LABEL_SCHEMA = PROJECT_ROOT / "training" / "label_schemas" / "motion_labels.json"
DEFAULT_PHASE_LABELS = ("none", "candidate", "starting", "active", "completed", "ending", "cancelled")
TRUTHY_VALUES = {"1", "true", "yes", "y"}
FALSY_VALUES = {"", "0", "false", "no", "n"}

DIRECTION_TO_ID = {
    "left": 0,
    "right": 1,
    "up": 2,
    "down": 3,
    "up_left": 4,
    "up_right": 5,
    "down_left": 6,
    "down_right": 7,
}
SCALE_LABEL_PREFIXES = ("ZOOM_", "PUSH", "PULL")
ROTATION_LABEL_PREFIXES = ("ROTATE_", "CIRCLE_")
DRAW_LABEL_PREFIX = "AIR_DRAW_"


@dataclass(frozen=True)
class MotionFrameRecord:
    row: Mapping[str, str]
    sample_id: str
    session_id: str
    frame_index: int
    elapsed_ms: float
    phase: str
    active_labels: frozenset[str]
    handedness_id: int
    landmarks: tuple[tuple[float, ...], ...]
    global_features: tuple[float, ...]
    landmark_mask: tuple[bool, ...]
    frame_mask: bool


@dataclass(frozen=True)
class MotionSequence:
    sample_id: str
    session_id: str
    target_name: str
    target_handedness: str
    target_speed: str
    frames: tuple[MotionFrameRecord, ...]


@dataclass(frozen=True)
class MotionWindowSpec:
    sequence_index: int
    end_frame_index: int


@dataclass(frozen=True)
class MotionWindowDatasetProfile:
    csv_path: Path
    total_samples: int
    total_frames: int
    total_windows: int
    label_counts: dict[str, int]
    phase_counts: dict[str, int]
    handedness_counts: dict[str, int]
    speed_counts: dict[str, int]
    missing_hand_frames: int
    padded_windows: int
    unlabeled_windows: int
    duplicate_frame_keys: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "csv_path": str(self.csv_path),
            "total_samples": self.total_samples,
            "total_frames": self.total_frames,
            "total_windows": self.total_windows,
            "label_counts": self.label_counts,
            "phase_counts": self.phase_counts,
            "handedness_counts": self.handedness_counts,
            "speed_counts": self.speed_counts,
            "missing_hand_frames": self.missing_hand_frames,
            "padded_windows": self.padded_windows,
            "unlabeled_windows": self.unlabeled_windows,
            "duplicate_frame_keys": list(self.duplicate_frame_keys),
        }

    def write_json(self, path: Path | str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as profile_file:
            json.dump(self.to_dict(), profile_file, indent=2, sort_keys=True)
            profile_file.write("\n")


class MotionWindowDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        csv_path: Path | str,
        *,
        label_schema_path: Path | str = DEFAULT_MOTION_LABEL_SCHEMA,
        feature_spec: LandmarkFeatureSpec | None = None,
        sequence_length: int = 32,
        window_stride: int = 4,
        min_interval_overlap: float = 0.25,
        endpoint_positive: bool = True,
        strict: bool = True,
        limit_samples: int | None = None,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero")
        if window_stride <= 0:
            raise ValueError("window_stride must be greater than zero")
        if min_interval_overlap < 0 or min_interval_overlap > 1:
            raise ValueError("min_interval_overlap must be between 0 and 1")
        self.csv_path = Path(csv_path)
        self.label_schema_payload = load_label_schema_payload(label_schema_path)
        self.label_schema = LabelSchema.from_mapping(self.label_schema_payload)
        if self.label_schema.model_family and self.label_schema.model_family != "motion":
            raise ValueError("motion window dataset requires a motion label schema")
        self.phase_labels = load_phase_labels(self.label_schema_payload)
        self.phase_to_index = {phase: index for index, phase in enumerate(self.phase_labels)}
        self.feature_spec = feature_spec or LandmarkFeatureSpec(global_feature_names=DEFAULT_MOTION_GLOBAL_FEATURES)
        self.sequence_length = sequence_length
        self.window_stride = window_stride
        self.min_interval_overlap = min_interval_overlap
        self.endpoint_positive = endpoint_positive
        self.strict = strict
        self.sequences, self.duplicate_frame_keys = self._load_sequences(limit_samples)
        if not self.sequences:
            raise ValueError(f"motion CSV contains no samples: {self.csv_path}")
        self.windows = build_window_specs(self.sequences, window_stride=self.window_stride)
        if not self.windows:
            raise ValueError(f"motion CSV produced no training windows: {self.csv_path}")

    @property
    def num_labels(self) -> int:
        return len(self.label_schema.labels)

    @property
    def num_phases(self) -> int:
        return len(self.phase_labels)

    @property
    def landmark_dim(self) -> int:
        return self.feature_spec.per_landmark_dim

    @property
    def global_feature_dim(self) -> int:
        return self.feature_spec.global_feature_dim

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        sequence = self.sequences[window.sequence_index]
        frames = sequence.frames
        end_index = window.end_frame_index
        start_index = max(0, end_index - self.sequence_length + 1)
        actual_frames = list(frames[start_index:end_index + 1])
        padding_count = self.sequence_length - len(actual_frames)
        padded_frames = [None for _ in range(padding_count)] + actual_frames
        endpoint_frame = frames[end_index]
        action_targets, phase_targets, phase_mask = build_window_targets(
            actual_frames,
            endpoint_frame,
            self.label_schema,
            self.phase_to_index,
            min_interval_overlap=self.min_interval_overlap,
            endpoint_positive=self.endpoint_positive,
        )
        label_vector = [action_targets[label] for label in self.label_schema.labels]
        positive_labels = {label for label, value in action_targets.items() if value > 0}

        return {
            "landmarks": torch.tensor(
                [
                    frame.landmarks if frame is not None else zero_landmarks(self.feature_spec)
                    for frame in padded_frames
                ],
                dtype=torch.float32,
            ),
            "global_features": torch.tensor(
                [
                    frame.global_features if frame is not None else zero_global_features(self.feature_spec)
                    for frame in padded_frames
                ],
                dtype=torch.float32,
            ),
            "handedness": torch.tensor(
                [frame.handedness_id if frame is not None else 0 for frame in padded_frames],
                dtype=torch.long,
            ),
            "frame_mask": torch.tensor(
                [bool(frame and frame.frame_mask) for frame in padded_frames],
                dtype=torch.bool,
            ),
            "landmark_mask": torch.tensor(
                [
                    frame.landmark_mask if frame is not None else zero_landmark_mask(self.feature_spec)
                    for frame in padded_frames
                ],
                dtype=torch.bool,
            ),
            "labels": torch.tensor(label_vector, dtype=torch.float32),
            "phase_targets": torch.tensor(phase_targets, dtype=torch.long),
            "phase_mask": torch.tensor(phase_mask, dtype=torch.bool),
            "aux": build_parameter_targets(actual_frames, endpoint_frame, positive_labels, self.feature_spec),
            "metadata": build_window_metadata(sequence, endpoint_frame, padding_count, len(actual_frames)),
        }

    def profile(self) -> MotionWindowDatasetProfile:
        return profile_sequences(
            self.csv_path,
            self.sequences,
            self.windows,
            self.label_schema,
            self.phase_labels,
            sequence_length=self.sequence_length,
            duplicate_frame_keys=self.duplicate_frame_keys,
        )

    def schema_payload(self) -> dict[str, object]:
        return dict(self.label_schema_payload)

    def _load_sequences(self, limit_samples: int | None) -> tuple[tuple[MotionSequence, ...], tuple[str, ...]]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"motion CSV not found: {self.csv_path}")
        grouped_rows: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        duplicate_frame_keys: Counter[str] = Counter()
        with self.csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise ValueError(f"motion CSV has no header: {self.csv_path}")
            validate_header(reader.fieldnames, self.label_schema, self.feature_spec)
            for row in reader:
                sample_id = (row.get("sample_id") or "").strip()
                if not sample_id:
                    if self.strict:
                        raise ValueError("motion CSV row is missing sample_id")
                    continue
                if limit_samples is not None and sample_id not in grouped_rows and len(grouped_rows) >= limit_samples:
                    continue
                frame_index = (row.get("sample_frame_index") or "").strip()
                duplicate_frame_keys.update([f"{sample_id}:{frame_index}"])
                grouped_rows.setdefault(sample_id, []).append(row)

        sequences = [self._build_sequence(sample_id, rows) for sample_id, rows in grouped_rows.items()]
        duplicates = tuple(sorted(key for key, count in duplicate_frame_keys.items() if count > 1))
        return tuple(sequences), duplicates

    def _build_sequence(self, sample_id: str, rows: list[dict[str, str]]) -> MotionSequence:
        rows = sorted(rows, key=lambda row: parse_int(row, "sample_frame_index"))
        frames = tuple(self._build_frame(row) for row in rows)
        first_row = rows[0]
        return MotionSequence(
            sample_id=sample_id,
            session_id=(first_row.get("session_id") or "").strip(),
            target_name=(first_row.get("target_name") or "").strip(),
            target_handedness=(first_row.get("target_handedness") or "unknown").strip().casefold(),
            target_speed=(first_row.get("target_speed") or "unknown").strip().casefold(),
            frames=frames,
        )

    def _build_frame(self, row: Mapping[str, str]) -> MotionFrameRecord:
        phase = (row.get("phase") or "none").strip().casefold()
        if phase not in self.phase_to_index:
            if self.strict:
                raise ValueError(f"unknown motion phase: {phase}")
            phase = "none"
        frame_mask = not parse_bool(row.get("tracking_missing")) and not row_has_missing_landmark(
            row,
            self.feature_spec,
        )
        landmark_mask = tuple(frame_mask for _ in range(self.feature_spec.num_landmarks))
        return MotionFrameRecord(
            row=row,
            sample_id=(row.get("sample_id") or "").strip(),
            session_id=(row.get("session_id") or "").strip(),
            frame_index=parse_int(row, "sample_frame_index"),
            elapsed_ms=parse_float(row, "sample_elapsed_ms"),
            phase=phase,
            active_labels=frozenset(parse_active_label_set(row, self.label_schema, strict=self.strict)),
            handedness_id=self.feature_spec.handedness_id(row.get("target_handedness")),
            landmarks=tuple(tuple(point) for point in parse_landmarks(row, self.feature_spec)),
            global_features=tuple(parse_global_features(row, self.feature_spec)),
            landmark_mask=landmark_mask,
            frame_mask=frame_mask,
        )


def load_label_schema_payload(path: Path | str) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as schema_file:
        payload = json.load(schema_file)
    if not isinstance(payload, dict):
        raise ValueError(f"motion label schema must be an object: {path}")
    return payload


def load_phase_labels(payload: Mapping[str, object]) -> tuple[str, ...]:
    raw_phase_labels = payload.get("phase_labels", DEFAULT_PHASE_LABELS)
    if not isinstance(raw_phase_labels, Sequence) or isinstance(raw_phase_labels, str):
        raise ValueError("motion label schema phase_labels must be an array")
    phase_labels = tuple(str(phase).strip().casefold() for phase in raw_phase_labels)
    if "none" not in phase_labels:
        raise ValueError("motion phase labels must include none")
    return phase_labels


def validate_header(fieldnames: Sequence[str], label_schema: LabelSchema, feature_spec: LandmarkFeatureSpec) -> None:
    present = set(fieldnames)
    required = {
        "sample_id",
        "session_id",
        "target_name",
        "target_handedness",
        "target_speed",
        "sample_frame_index",
        "sample_elapsed_ms",
        "phase",
        "active_labels",
        "tracking_missing",
        *feature_spec.landmark_columns(),
        *feature_spec.global_feature_names,
    }
    required.update(f"label_{label}" for label in label_schema.labels)
    missing = sorted(required - present)
    if missing:
        preview = ", ".join(missing[:12])
        suffix = "..." if len(missing) > 12 else ""
        raise ValueError(f"motion CSV is missing required columns: {preview}{suffix}")


def build_window_specs(sequences: Sequence[MotionSequence], *, window_stride: int) -> tuple[MotionWindowSpec, ...]:
    windows: list[MotionWindowSpec] = []
    for sequence_index, sequence in enumerate(sequences):
        if not sequence.frames:
            continue
        end_indices = list(range(0, len(sequence.frames), window_stride))
        if end_indices[-1] != len(sequence.frames) - 1:
            end_indices.append(len(sequence.frames) - 1)
        windows.extend(MotionWindowSpec(sequence_index, end_index) for end_index in end_indices)
    return tuple(windows)


def build_window_targets(
    frames: Sequence[MotionFrameRecord],
    endpoint_frame: MotionFrameRecord,
    label_schema: LabelSchema,
    phase_to_index: Mapping[str, int],
    *,
    min_interval_overlap: float,
    endpoint_positive: bool,
) -> tuple[dict[str, float], list[int], list[bool]]:
    action_targets: dict[str, float] = {label: 0.0 for label in label_schema.labels}
    phase_targets = [phase_to_index["none"] for _ in label_schema.labels]
    phase_mask = [False for _ in label_schema.labels]
    label_counts = Counter(label for frame in frames for label in frame.active_labels)
    frame_count = max(1, len(frames))
    for label_index, label in enumerate(label_schema.labels):
        overlap_positive = label_counts[label] / frame_count >= min_interval_overlap
        endpoint_has_label = label in endpoint_frame.active_labels
        if (endpoint_positive and endpoint_has_label) or overlap_positive:
            action_targets[label] = 1.0
            phase = phase_for_label(label, endpoint_frame, frames)
            phase_targets[label_index] = phase_to_index.get(phase, phase_to_index["none"])
            phase_mask[label_index] = True
    return action_targets, phase_targets, phase_mask


def phase_for_label(label: str, endpoint_frame: MotionFrameRecord, frames: Sequence[MotionFrameRecord]) -> str:
    if label in endpoint_frame.active_labels:
        return endpoint_frame.phase
    for frame in reversed(frames):
        if label in frame.active_labels:
            return frame.phase
    return "none"


def build_parameter_targets(
    frames: Sequence[MotionFrameRecord],
    endpoint_frame: MotionFrameRecord,
    action_targets: set[str],
    feature_spec: LandmarkFeatureSpec,
) -> dict[str, torch.Tensor]:
    first_valid = next((frame for frame in frames if frame.frame_mask), frames[0]) if frames else endpoint_frame
    current_palm = vector_from_global(endpoint_frame, feature_spec, "feature_palm_center", 3)
    first_palm = vector_from_global(first_valid, feature_spec, "feature_palm_center", 3)
    displacement = [current_palm[index] - first_palm[index] for index in range(3)]
    velocity = vector_from_global(endpoint_frame, feature_spec, "feature_velocity", 3)
    scale_delta = global_feature(endpoint_frame, feature_spec, "feature_scale") - global_feature(
        first_valid,
        feature_spec,
        "feature_scale",
    )
    rotation_delta = hand_orientation_degrees(endpoint_frame) - hand_orientation_degrees(first_valid)
    direction_target = direction_id_for_labels(action_targets)
    any_action = bool(action_targets)
    return {
        "direction": torch.tensor(direction_target if direction_target is not None else 0, dtype=torch.long),
        "direction_mask": torch.tensor(direction_target is not None, dtype=torch.bool),
        "displacement": torch.tensor(displacement, dtype=torch.float32),
        "displacement_mask": torch.tensor(any_action, dtype=torch.bool),
        "velocity": torch.tensor(velocity, dtype=torch.float32),
        "velocity_mask": torch.tensor(any_action, dtype=torch.bool),
        "duration_ms": torch.tensor(duration_target_ms(endpoint_frame), dtype=torch.float32),
        "duration_mask": torch.tensor(any_action, dtype=torch.bool),
        "scale_delta": torch.tensor(scale_delta, dtype=torch.float32),
        "scale_mask": torch.tensor(
            any(label.startswith(SCALE_LABEL_PREFIXES) for label in action_targets),
            dtype=torch.bool,
        ),
        "rotation_degrees": torch.tensor(rotation_delta, dtype=torch.float32),
        "rotation_mask": torch.tensor(
            any(label.startswith(ROTATION_LABEL_PREFIXES) for label in action_targets),
            dtype=torch.bool,
        ),
        "draw_point": torch.tensor(draw_point_target(endpoint_frame), dtype=torch.float32),
        "draw_mask": torch.tensor(
            any(label.startswith(DRAW_LABEL_PREFIX) for label in action_targets),
            dtype=torch.bool,
        ),
    }


def build_window_metadata(
    sequence: MotionSequence,
    endpoint_frame: MotionFrameRecord,
    padding_count: int,
    actual_frame_count: int,
) -> dict[str, object]:
    return {
        "sample_id": sequence.sample_id,
        "session_id": sequence.session_id,
        "target_name": sequence.target_name,
        "target_handedness": sequence.target_handedness,
        "target_speed": sequence.target_speed,
        "endpoint_frame_index": endpoint_frame.frame_index,
        "endpoint_elapsed_ms": endpoint_frame.elapsed_ms,
        "endpoint_phase": endpoint_frame.phase,
        "padding_count": padding_count,
        "actual_frame_count": actual_frame_count,
    }


def parse_active_label_set(row: Mapping[str, str], label_schema: LabelSchema, *, strict: bool) -> set[str]:
    labels: set[str] = set()
    active_labels_value = (row.get("active_labels") or "").strip()
    if active_labels_value:
        labels.update(label.strip() for label in active_labels_value.split("|") if label.strip())
    for label in label_schema.labels:
        value = (row.get(f"label_{label}") or "").strip().casefold()
        if value in TRUTHY_VALUES:
            labels.add(label)
        elif value not in FALSY_VALUES and strict:
            raise ValueError(f"invalid binary value for label_{label}: {value!r}")
    unknown = sorted(labels - set(label_schema.labels))
    if unknown and strict:
        raise ValueError(f"row contains motion labels not present in schema: {', '.join(unknown)}")
    return labels - set(unknown)


def parse_landmarks(row: Mapping[str, str], feature_spec: LandmarkFeatureSpec) -> list[list[float]]:
    landmarks: list[list[float]] = []
    for landmark_index in range(feature_spec.num_landmarks):
        landmarks.append(
            [
                parse_float(row, feature_spec.landmark_column(landmark_index, axis))
                for axis in feature_spec.landmark_axes
            ]
        )
    return landmarks


def parse_global_features(row: Mapping[str, str], feature_spec: LandmarkFeatureSpec) -> list[float]:
    return [parse_float(row, feature_name) for feature_name in feature_spec.global_feature_names]


def parse_float(row: Mapping[str, str], column: str) -> float:
    value = (row.get(column) or "").strip()
    if value == "":
        return 0.0
    try:
        value_float = float(value)
    except ValueError as exc:
        raise ValueError(f"invalid numeric value for {column}: {value!r}") from exc
    if not math.isfinite(value_float):
        raise ValueError(f"non-finite numeric value for {column}: {value!r}")
    return value_float


def parse_int(row: Mapping[str, str], column: str) -> int:
    return int(parse_float(row, column))


def parse_bool(value: str | None) -> bool:
    normalized = (value or "").strip().casefold()
    return normalized in TRUTHY_VALUES


def row_has_missing_landmark(row: Mapping[str, str], feature_spec: LandmarkFeatureSpec) -> bool:
    return any((row.get(column) or "").strip() == "" for column in feature_spec.landmark_columns())


def zero_landmarks(feature_spec: LandmarkFeatureSpec) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(0.0 for _ in feature_spec.landmark_axes) for _ in range(feature_spec.num_landmarks))


def zero_landmark_mask(feature_spec: LandmarkFeatureSpec) -> tuple[bool, ...]:
    return tuple(False for _ in range(feature_spec.num_landmarks))


def zero_global_features(feature_spec: LandmarkFeatureSpec) -> tuple[float, ...]:
    return tuple(0.0 for _ in feature_spec.global_feature_names)


def direction_id_for_labels(labels: Iterable[str]) -> int | None:
    for label in sorted(labels):
        direction = direction_suffix(label)
        if direction in DIRECTION_TO_ID:
            return DIRECTION_TO_ID[direction]
    return None


def direction_suffix(label: str) -> str | None:
    normalized = label.casefold()
    for direction in sorted(DIRECTION_TO_ID, key=len, reverse=True):
        if normalized.endswith(direction):
            return direction
    return None


def global_feature(frame: MotionFrameRecord, feature_spec: LandmarkFeatureSpec, name: str) -> float:
    feature_names = tuple(feature_spec.global_feature_names)
    if name not in feature_names:
        return 0.0
    return frame.global_features[feature_names.index(name)]


def vector_from_global(
    frame: MotionFrameRecord,
    feature_spec: LandmarkFeatureSpec,
    prefix: str,
    dim: int,
) -> list[float]:
    axes = ("x", "y", "z")[:dim]
    return [global_feature(frame, feature_spec, f"{prefix}_{axis}") for axis in axes]


def duration_target_ms(frame: MotionFrameRecord) -> float:
    if frame.phase == "none":
        return 0.0
    interval_start_time = parse_float(frame.row, "interval_start_time") * 1000.0
    return max(0.0, frame.elapsed_ms - interval_start_time)


def hand_orientation_degrees(frame: MotionFrameRecord) -> float:
    if len(frame.landmarks) <= 9:
        return 0.0
    wrist = frame.landmarks[0]
    middle_mcp = frame.landmarks[9]
    return math.degrees(math.atan2(middle_mcp[1] - wrist[1], middle_mcp[0] - wrist[0]))


def draw_point_target(frame: MotionFrameRecord) -> list[float]:
    if len(frame.landmarks) <= 8:
        return [0.0, 0.0]
    index_tip = frame.landmarks[8]
    return [index_tip[0], index_tip[1]]


def motion_window_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "landmarks": torch.stack([sample["landmarks"] for sample in batch]),
        "global_features": torch.stack([sample["global_features"] for sample in batch]),
        "handedness": torch.stack([sample["handedness"] for sample in batch]),
        "frame_mask": torch.stack([sample["frame_mask"] for sample in batch]),
        "landmark_mask": torch.stack([sample["landmark_mask"] for sample in batch]),
        "labels": torch.stack([sample["labels"] for sample in batch]),
        "phase_targets": torch.stack([sample["phase_targets"] for sample in batch]),
        "phase_mask": torch.stack([sample["phase_mask"] for sample in batch]),
        "aux": collate_auxiliary_targets([sample["aux"] for sample in batch]),
        "metadata": [sample["metadata"] for sample in batch],
    }


def collate_auxiliary_targets(aux_values: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = aux_values[0].keys()
    return {key: torch.stack([aux[key] for aux in aux_values]) for key in keys}


def split_motion_window_indices(
    dataset: MotionWindowDataset,
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 1337,
) -> dict[str, list[int]]:
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must be non-negative and sum to less than one")
    sample_ids = [sequence.sample_id for sequence in dataset.sequences]
    random.Random(seed).shuffle(sample_ids)
    test_count = int(round(len(sample_ids) * test_fraction))
    val_count = int(round(len(sample_ids) * val_fraction))
    if len(sample_ids) >= 3 and val_fraction > 0:
        val_count = max(1, val_count)
    if len(sample_ids) >= 3 and test_fraction > 0:
        test_count = max(1, test_count)
    if val_count + test_count >= len(sample_ids):
        overflow = val_count + test_count - len(sample_ids) + 1
        val_count = max(0, val_count - overflow)
    test_ids = set(sample_ids[:test_count])
    val_ids = set(sample_ids[test_count:test_count + val_count])
    train_ids = set(sample_ids[test_count + val_count:])
    if not train_ids:
        raise ValueError("split configuration produced an empty training split")
    split_indices = {"train": [], "validation": [], "test": []}
    for window_index, window in enumerate(dataset.windows):
        sample_id = dataset.sequences[window.sequence_index].sample_id
        if sample_id in test_ids:
            split_indices["test"].append(window_index)
        elif sample_id in val_ids:
            split_indices["validation"].append(window_index)
        else:
            split_indices["train"].append(window_index)
    return {split: indices for split, indices in split_indices.items() if indices}


def build_motion_window_dataloaders(
    csv_path: Path | str,
    *,
    label_schema_path: Path | str = DEFAULT_MOTION_LABEL_SCHEMA,
    feature_spec: LandmarkFeatureSpec | None = None,
    sequence_length: int = 32,
    window_stride: int = 4,
    min_interval_overlap: float = 0.25,
    endpoint_positive: bool = True,
    batch_size: int = 32,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 1337,
    num_workers: int = 0,
    limit_samples: int | None = None,
    pin_memory: bool = False,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
) -> tuple[dict[str, DataLoader[dict[str, Any]]], MotionWindowDataset, dict[str, list[int]]]:
    dataset = MotionWindowDataset(
        csv_path,
        label_schema_path=label_schema_path,
        feature_spec=feature_spec,
        sequence_length=sequence_length,
        window_stride=window_stride,
        min_interval_overlap=min_interval_overlap,
        endpoint_positive=endpoint_positive,
        limit_samples=limit_samples,
    )
    split_indices = split_motion_window_indices(
        dataset,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    dataloaders: dict[str, DataLoader[dict[str, Any]]] = {}
    for split_name, indices in split_indices.items():
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
            collate_fn=motion_window_collate,
            pin_memory=pin_memory,
        )
    return dataloaders, dataset, split_indices


def compute_motion_pos_weight(dataset: Dataset[dict[str, Any]]) -> torch.Tensor:
    labels = [dataset[index]["labels"] for index in range(len(dataset))]
    stacked = torch.stack(labels)
    positives = stacked.sum(dim=0)
    negatives = stacked.shape[0] - positives
    return torch.where(positives > 0, negatives / positives.clamp_min(1.0), torch.ones_like(positives))


def profile_motion_csv(
    csv_path: Path | str,
    *,
    label_schema_path: Path | str = DEFAULT_MOTION_LABEL_SCHEMA,
    feature_spec: LandmarkFeatureSpec | None = None,
    sequence_length: int = 32,
    window_stride: int = 4,
) -> MotionWindowDatasetProfile:
    dataset = MotionWindowDataset(
        csv_path,
        label_schema_path=label_schema_path,
        feature_spec=feature_spec,
        sequence_length=sequence_length,
        window_stride=window_stride,
    )
    return dataset.profile()


def profile_sequences(
    csv_path: Path,
    sequences: Sequence[MotionSequence],
    windows: Sequence[MotionWindowSpec],
    label_schema: LabelSchema,
    phase_labels: Sequence[str],
    *,
    sequence_length: int,
    duplicate_frame_keys: Sequence[str],
) -> MotionWindowDatasetProfile:
    label_counts = Counter({label: 0 for label in label_schema.labels})
    phase_counts = Counter({phase: 0 for phase in phase_labels})
    handedness_counts: Counter[str] = Counter()
    speed_counts: Counter[str] = Counter()
    total_frames = 0
    missing_hand_frames = 0
    for sequence in sequences:
        handedness_counts.update([sequence.target_handedness])
        speed_counts.update([sequence.target_speed])
        for frame in sequence.frames:
            total_frames += 1
            missing_hand_frames += int(not frame.frame_mask)
            label_counts.update(frame.active_labels)
            phase_counts.update([frame.phase])
    padded_windows = sum(
        max(0, sequence_length - (window.end_frame_index + 1))
        for window in windows
    )
    unlabeled_windows = sum(
        1 for window in windows if not sequences[window.sequence_index].frames[window.end_frame_index].active_labels
    )
    return MotionWindowDatasetProfile(
        csv_path=csv_path,
        total_samples=len(sequences),
        total_frames=total_frames,
        total_windows=len(windows),
        label_counts=dict(label_counts),
        phase_counts=dict(phase_counts),
        handedness_counts=dict(handedness_counts),
        speed_counts=dict(speed_counts),
        missing_hand_frames=missing_hand_frames,
        padded_windows=padded_windows,
        unlabeled_windows=unlabeled_windows,
        duplicate_frame_keys=tuple(duplicate_frame_keys),
    )
