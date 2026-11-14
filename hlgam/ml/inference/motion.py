from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from hlgam import settings
from hlgam.gestures.base import clamp_confidence
from hlgam.ml.features import global_feature_vector, landmark_matrix
from hlgam.ml.inference.static_pose import resolve_device
from hlgam.ml.schemas import LandmarkFeatureSpec, LabelSchema
from hlgam.models import GestureEvent, GestureFeatures, GesturePhase, Point3D


DEFAULT_MOTION_PHASE_LABELS = ("none", "candidate", "starting", "active", "completed", "ending", "cancelled")
DIRECTION_LABELS = ("left", "right", "up", "down", "up_left", "up_right", "down_left", "down_right")


@dataclass(frozen=True)
class MotionModelConfig:
    enabled: bool = False


@dataclass(frozen=True)
class MotionModelRuntimeSettings:
    checkpoint_path: Path
    label_schema_path: Path
    feature_schema_path: Path
    device: str
    threshold: float
    min_sequence_frames: int
    min_hold_ms: int
    cooldown_ms: int
    model_id: str | None = None


@dataclass(frozen=True)
class LoadedMotionModel:
    model: Any
    label_schema: LabelSchema
    phase_labels: tuple[str, ...]
    feature_spec: LandmarkFeatureSpec
    device: Any
    model_id: str
    checkpoint_path: Path
    sequence_length: int


@dataclass(frozen=True)
class MotionPrediction:
    label: str
    score: float
    phase_label: str
    phase_score: float
    direction: str | None
    quality_score: float | None
    displacement: Point3D | None
    velocity: Point3D | None
    duration_ms: float | None
    scale_delta: float | None
    rotation_degrees: float | None
    draw_point: dict[str, float] | None


class LearnedMotionRecognizer:
    name = "learned_motion"
    metadata_source = "learned_motion"

    def __init__(self, config: MotionModelConfig | None = None) -> None:
        self.config = config or MotionModelConfig()
        self._settings: MotionModelRuntimeSettings | None = None
        self._loaded: LoadedMotionModel | None = None
        self._history: deque[GestureFeatures] = deque(maxlen=2)
        self._active_since: dict[str, float] = {}
        self._cooldown_until: dict[str, float] = {}

    def process(self, features: GestureFeatures) -> list[GestureEvent]:
        if not self.config.enabled:
            return []
        settings = self._ensure_settings()
        loaded = self._ensure_loaded(settings)
        if self._history.maxlen != loaded.sequence_length:
            self._history = deque(self._history, maxlen=max(1, loaded.sequence_length))
        self._history.append(features)
        if len(self._history) < settings.min_sequence_frames:
            return []

        predictions = {
            prediction.label: prediction
            for prediction in self._predict(features, loaded)
            if prediction.score >= settings.threshold
        }
        for label in tuple(self._active_since):
            if label not in predictions:
                self._active_since.pop(label, None)

        timestamp = features.timestamp
        events: list[GestureEvent] = []
        for label in sorted(predictions):
            active_since = self._active_since.setdefault(label, timestamp)
            held_ms = (timestamp - active_since) * 1000.0
            if held_ms < settings.min_hold_ms:
                continue
            if timestamp < self._cooldown_until.get(label, 0.0):
                continue
            self._cooldown_until[label] = timestamp + (settings.cooldown_ms / 1000.0)
            events.append(self._event(predictions[label], features, loaded, settings, held_ms))
        return events

    def reset(self) -> None:
        self._history.clear()
        self._active_since.clear()
        self._cooldown_until.clear()

    def _ensure_settings(self) -> MotionModelRuntimeSettings:
        if self._settings is None:
            self._settings = load_motion_runtime_settings()
        return self._settings

    def _ensure_loaded(self, settings: MotionModelRuntimeSettings) -> LoadedMotionModel:
        if self._loaded is None:
            self._loaded = load_motion_model(settings)
        return self._loaded

    def _predict(self, features: GestureFeatures, loaded: LoadedMotionModel) -> list[MotionPrediction]:
        import torch

        window = list(self._history)[-loaded.sequence_length:]
        padding_count = loaded.sequence_length - len(window)
        padded_window: list[GestureFeatures | None] = [None for _ in range(padding_count)] + window
        landmarks = torch.tensor(
            [[_landmarks_for_frame(frame, loaded.feature_spec) for frame in padded_window]],
            dtype=torch.float32,
            device=loaded.device,
        )
        global_features = torch.tensor(
            [[_global_features_for_frame(frame, loaded.feature_spec) for frame in padded_window]],
            dtype=torch.float32,
            device=loaded.device,
        )
        handedness = torch.tensor(
            [[_handedness_for_frame(frame, loaded.feature_spec) for frame in padded_window]],
            dtype=torch.long,
            device=loaded.device,
        )
        frame_mask = torch.tensor(
            [[frame is not None for frame in padded_window]],
            dtype=torch.bool,
            device=loaded.device,
        )
        landmark_mask = torch.tensor(
            [[_landmark_mask_for_frame(frame, loaded.feature_spec) for frame in padded_window]],
            dtype=torch.bool,
            device=loaded.device,
        )
        with torch.inference_mode():
            outputs = loaded.model(
                {
                    "landmarks": landmarks,
                    "global_features": global_features,
                    "handedness": handedness,
                    "frame_mask": frame_mask,
                    "landmark_mask": landmark_mask,
                }
            )
            action_scores = torch.sigmoid(outputs["action_logits"])[0].detach().cpu().tolist()
            phase_scores = torch.softmax(outputs["phase_logits"], dim=-1)[0].detach().cpu().tolist()
            direction_id = int(outputs["direction_logits"][0].argmax(dim=-1).detach().cpu().item())
            displacement = _tensor_vector(outputs.get("displacement"), torch)
            velocity = _tensor_vector(outputs.get("velocity"), torch)
            duration_ms = _tensor_scalar(outputs.get("duration_ms"), torch)
            scale_delta = _tensor_scalar(outputs.get("scale_delta"), torch)
            rotation_degrees = _tensor_scalar(outputs.get("rotation_degrees"), torch)
            draw_point = _tensor_vector(outputs.get("draw_point"), torch)
            quality_score = _quality_score(outputs.get("quality_logit"), torch)

        direction = DIRECTION_LABELS[direction_id] if 0 <= direction_id < len(DIRECTION_LABELS) else None
        predictions: list[MotionPrediction] = []
        for index, label in enumerate(loaded.label_schema.labels):
            label_phase_scores = phase_scores[index]
            phase_index = max(range(len(label_phase_scores)), key=label_phase_scores.__getitem__)
            phase_label = loaded.phase_labels[phase_index] if phase_index < len(loaded.phase_labels) else "none"
            predictions.append(
                MotionPrediction(
                    label=label,
                    score=clamp_confidence(float(action_scores[index])),
                    phase_label=phase_label,
                    phase_score=clamp_confidence(float(label_phase_scores[phase_index])),
                    direction=direction,
                    quality_score=quality_score,
                    displacement=_point_from_vector(displacement),
                    velocity=_point_from_vector(velocity),
                    duration_ms=duration_ms,
                    scale_delta=scale_delta,
                    rotation_degrees=rotation_degrees,
                    draw_point=_draw_point_payload(draw_point),
                )
            )
        return predictions

    def _event(
        self,
        prediction: MotionPrediction,
        features: GestureFeatures,
        loaded: LoadedMotionModel,
        settings: MotionModelRuntimeSettings,
        held_ms: float,
    ) -> GestureEvent:
        metadata: dict[str, object] = {
            "source": self.metadata_source,
            "model_id": loaded.model_id,
            "model_score": prediction.score,
            "threshold": settings.threshold,
            "held_ms": held_ms,
            "handedness": features.landmarks.handedness,
            "model_phase": prediction.phase_label,
            "model_phase_score": prediction.phase_score,
            "sequence_frames": min(len(self._history), loaded.sequence_length),
        }
        if prediction.direction is not None:
            metadata["direction"] = prediction.direction
        if prediction.quality_score is not None:
            metadata["quality_score"] = prediction.quality_score
        if prediction.displacement is not None:
            metadata["displacement"] = prediction.displacement.to_dict()
        if prediction.velocity is not None:
            metadata["predicted_velocity"] = prediction.velocity.to_dict()
        if prediction.duration_ms is not None:
            metadata["duration_ms"] = prediction.duration_ms
        if prediction.scale_delta is not None:
            metadata["scale_delta"] = prediction.scale_delta
        if prediction.rotation_degrees is not None:
            metadata["rotation_degrees"] = prediction.rotation_degrees
        if prediction.draw_point is not None:
            metadata["draw_point"] = prediction.draw_point
        return GestureEvent(
            gesture=prediction.label,
            phase=gesture_phase_for_motion(prediction.label, prediction.phase_label),
            confidence=prediction.score,
            timestamp=features.timestamp,
            position=features.position_payload(),
            velocity=features.velocity_payload(),
            metadata=metadata,
        )


def load_motion_model(settings: MotionModelRuntimeSettings) -> LoadedMotionModel:
    try:
        import torch
        from hlgam.ml.models.motion_transformer import MotionGestureTransformer, MotionGestureTransformerConfig
    except ImportError as exc:
        raise RuntimeError("motion model inference requires PyTorch; install the ml extra") from exc

    checkpoint_path = settings.checkpoint_path
    device = resolve_device(settings.device, torch)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"motion checkpoint is missing model_state_dict: {checkpoint_path}")

    label_schema_payload = _load_json_mapping(settings.label_schema_path)
    label_schema = LabelSchema.from_mapping(label_schema_payload)
    if label_schema.model_family and label_schema.model_family != "motion":
        raise RuntimeError("motion model requires a motion label schema")
    phase_labels = _phase_labels_from_schema(label_schema_payload)
    feature_spec = LandmarkFeatureSpec.from_mapping(_load_json_mapping(settings.feature_schema_path))
    training_config = checkpoint.get("config", {})
    model_section = training_config.get("model", {}) if isinstance(training_config, Mapping) else {}
    data_section = training_config.get("data", {}) if isinstance(training_config, Mapping) else {}
    sequence_length = int(
        model_section.get(
            "max_sequence_length",
            data_section.get("sequence_length", _infer_sequence_length(checkpoint, default=32)),
        )
    )
    model_config = MotionGestureTransformerConfig(
        input_dim=feature_spec.per_landmark_dim,
        global_feature_dim=feature_spec.global_feature_dim,
        num_labels=len(label_schema.labels),
        num_phases=len(phase_labels),
        spatial_d_model=int(model_section.get("spatial_d_model", 128)),
        spatial_layers=int(model_section.get("spatial_layers", 2)),
        spatial_heads=int(model_section.get("spatial_heads", 4)),
        spatial_ffn_dim=int(model_section.get("spatial_ffn_dim", 512)),
        temporal_d_model=int(model_section.get("temporal_d_model", 256)),
        temporal_layers=int(model_section.get("temporal_layers", 4)),
        temporal_heads=int(model_section.get("temporal_heads", 8)),
        temporal_ffn_dim=int(model_section.get("temporal_ffn_dim", 1024)),
        max_sequence_length=max(1, sequence_length),
        dropout=float(model_section.get("dropout", 0.1)),
        use_global_features=bool(model_section.get("use_global_features", True)),
        causal=bool(model_section.get("causal", False)),
    )
    model = MotionGestureTransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    model_id = str(settings.model_id or checkpoint_path.parent.parent.name or checkpoint_path.stem)
    return LoadedMotionModel(
        model=model,
        label_schema=label_schema,
        phase_labels=phase_labels,
        feature_spec=feature_spec,
        device=device,
        model_id=model_id,
        checkpoint_path=checkpoint_path,
        sequence_length=model_config.max_sequence_length,
    )


def load_motion_runtime_settings(env: Mapping[str, str] | None = None) -> MotionModelRuntimeSettings:
    runtime_env = settings.motion_model_env(env)
    return MotionModelRuntimeSettings(
        checkpoint_path=runtime_env.checkpoint_path,
        label_schema_path=runtime_env.label_schema_path,
        feature_schema_path=runtime_env.feature_schema_path,
        device=runtime_env.device,
        threshold=runtime_env.threshold,
        min_sequence_frames=runtime_env.min_sequence_frames,
        min_hold_ms=runtime_env.min_hold_ms,
        cooldown_ms=runtime_env.cooldown_ms,
        model_id=runtime_env.model_id,
    )


def gesture_phase_for_motion(label: str, phase_label: str) -> GesturePhase:
    normalized_phase = phase_label.casefold()
    if normalized_phase == "starting" or label.endswith("_START"):
        return GesturePhase.STARTED
    if normalized_phase in {"active", "candidate"} or label.endswith(("_MOVE", "_HOLD")):
        return GesturePhase.UPDATED
    if normalized_phase == "ending" or label.endswith("_END"):
        return GesturePhase.ENDED
    if normalized_phase == "cancelled":
        return GesturePhase.CANCELLED
    return GesturePhase.COMPLETED


def _load_json_mapping(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as schema_file:
        payload = json.load(schema_file)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"schema must be a JSON object: {path}")
    return payload


def _phase_labels_from_schema(payload: Mapping[str, object]) -> tuple[str, ...]:
    raw_labels = payload.get("phase_labels", DEFAULT_MOTION_PHASE_LABELS)
    if not isinstance(raw_labels, list | tuple):
        raise RuntimeError("motion label schema phase_labels must be an array")
    phase_labels = tuple(str(label).strip().casefold() for label in raw_labels)
    if "none" not in phase_labels:
        raise RuntimeError("motion phase labels must include none")
    return phase_labels


def _infer_sequence_length(checkpoint: Mapping[str, Any], *, default: int) -> int:
    state_dict = checkpoint.get("model_state_dict", {})
    if isinstance(state_dict, Mapping):
        position_embedding = state_dict.get("temporal_encoder.position_embedding")
        if (
            position_embedding is not None
            and hasattr(position_embedding, "shape")
            and len(position_embedding.shape) >= 2
        ):
            return int(position_embedding.shape[1])
    return default


def _landmarks_for_frame(frame: GestureFeatures | None, feature_spec: LandmarkFeatureSpec) -> list[list[float]]:
    if frame is None:
        return [[0.0 for _ in feature_spec.landmark_axes] for _ in range(feature_spec.num_landmarks)]
    return landmark_matrix(frame, feature_spec)


def _global_features_for_frame(frame: GestureFeatures | None, feature_spec: LandmarkFeatureSpec) -> list[float]:
    if frame is None:
        return [0.0 for _ in feature_spec.global_feature_names]
    return global_feature_vector(frame, feature_spec)


def _handedness_for_frame(frame: GestureFeatures | None, feature_spec: LandmarkFeatureSpec) -> int:
    if frame is None:
        return feature_spec.handedness_id(None)
    return feature_spec.handedness_id(frame.landmarks.handedness)


def _landmark_mask_for_frame(frame: GestureFeatures | None, feature_spec: LandmarkFeatureSpec) -> list[bool]:
    return [frame is not None for _ in range(feature_spec.num_landmarks)]


def _tensor_vector(value: Any, torch_module: Any) -> list[float] | None:
    if value is None:
        return None
    return [float(item) for item in value[0].detach().cpu().reshape(-1).tolist()]


def _tensor_scalar(value: Any, torch_module: Any) -> float | None:
    if value is None:
        return None
    return float(value[0].detach().cpu().reshape(-1)[0].item())


def _quality_score(value: Any, torch_module: Any) -> float | None:
    scalar = _tensor_scalar(value, torch_module)
    if scalar is None:
        return None
    return clamp_confidence(float(torch_module.sigmoid(torch_module.tensor(scalar)).item()))


def _point_from_vector(value: list[float] | None) -> Point3D | None:
    if value is None or len(value) < 3:
        return None
    return Point3D(float(value[0]), float(value[1]), float(value[2]))


def _draw_point_payload(value: list[float] | None) -> dict[str, float] | None:
    if value is None or len(value) < 2:
        return None
    return {"x": float(value[0]), "y": float(value[1])}
