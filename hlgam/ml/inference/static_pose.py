from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from hlgam import settings
from hlgam.ml.features import global_feature_vector, landmark_matrix
from hlgam.ml.schemas import LandmarkFeatureSpec, LabelSchema
from hlgam.models import GestureEvent, GestureFeatures, GesturePhase


STATIC_POSE_MODEL_GROUP_LABELS: Mapping[str, frozenset[str]] = {
    "pose": frozenset(
        {
            "OPEN_PALM",
            "FIST",
            "POINT_INDEX",
            "TWO_FINGER_POINT",
            "PEACE_SIGN",
            "OK_SIGN",
            "THUMBS_UP",
            "THUMBS_DOWN",
            "L_SHAPE",
            "ROCK_SIGN",
            "HANG_LOOSE",
        }
    ),
    "finger_count": frozenset(
        {
            "FINGER_COUNT_1",
            "FINGER_COUNT_2",
            "FINGER_COUNT_3",
            "FINGER_COUNT_4",
            "FINGER_COUNT_5",
        }
    ),
    "finger_pointing": frozenset(
        {
            "FINGER_POINT_LEFT",
            "FINGER_POINT_RIGHT",
            "FINGER_POINT_UP",
            "FINGER_POINT_DOWN",
            "FINGER_POINT_UP_LEFT",
            "FINGER_POINT_UP_RIGHT",
            "FINGER_POINT_DOWN_LEFT",
            "FINGER_POINT_DOWN_RIGHT",
        }
    ),
    "flat_hand": frozenset(
        {
            "FLAT_HAND_LEFT",
            "FLAT_HAND_RIGHT",
            "FLAT_HAND_UP",
            "FLAT_HAND_DOWN",
        }
    ),
}


@dataclass(frozen=True)
class StaticPoseModelGroupConfig:
    enabled: bool = True


@dataclass(frozen=True)
class StaticPoseModelConfig:
    enabled: bool = False
    pose: StaticPoseModelGroupConfig = StaticPoseModelGroupConfig()
    finger_count: StaticPoseModelGroupConfig = StaticPoseModelGroupConfig()
    finger_pointing: StaticPoseModelGroupConfig = StaticPoseModelGroupConfig()
    flat_hand: StaticPoseModelGroupConfig = StaticPoseModelGroupConfig()

    def enabled_labels(self) -> frozenset[str]:
        labels: set[str] = set()
        for group_name, group_labels in STATIC_POSE_MODEL_GROUP_LABELS.items():
            if getattr(self, group_name).enabled:
                labels.update(group_labels)
        return frozenset(labels)


@dataclass(frozen=True)
class StaticPoseModelRuntimeSettings:
    checkpoint_path: Path
    label_schema_path: Path
    feature_schema_path: Path
    device: str
    threshold: float
    min_hold_ms: int
    cooldown_ms: int
    model_id: str | None = None


@dataclass(frozen=True)
class LoadedStaticPoseModel:
    model: Any
    label_schema: LabelSchema
    feature_spec: LandmarkFeatureSpec
    device: Any
    model_id: str
    checkpoint_path: Path


class LearnedStaticPoseRecognizer:
    name = "learned_static_pose"
    metadata_source = "learned_static_pose"

    def __init__(self, config: StaticPoseModelConfig | None = None) -> None:
        self.config = config or StaticPoseModelConfig()
        self._settings: StaticPoseModelRuntimeSettings | None = None
        self._loaded: LoadedStaticPoseModel | None = None
        self._active_since: dict[str, float] = {}
        self._cooldown_until: dict[str, float] = {}

    def process(self, features: GestureFeatures) -> list[GestureEvent]:
        if not self.config.enabled:
            return []
        enabled_labels = self.config.enabled_labels()
        if not enabled_labels:
            return []
        settings = self._ensure_settings()
        loaded = self._ensure_loaded(settings)
        scores = self._predict_scores(features, loaded)
        timestamp = features.timestamp
        candidates = {
            label: score
            for label, score in scores.items()
            if label in enabled_labels and score >= settings.threshold
        }
        for label in tuple(self._active_since):
            if label not in candidates:
                self._active_since.pop(label, None)

        events: list[GestureEvent] = []
        for label in sorted(candidates):
            active_since = self._active_since.setdefault(label, timestamp)
            held_ms = (timestamp - active_since) * 1000.0
            if held_ms < settings.min_hold_ms:
                continue
            if timestamp < self._cooldown_until.get(label, 0.0):
                continue
            self._cooldown_until[label] = timestamp + (settings.cooldown_ms / 1000.0)
            score = candidates[label]
            events.append(
                GestureEvent(
                    gesture=label,
                    phase=GesturePhase.COMPLETED,
                    confidence=score,
                    timestamp=timestamp,
                    position=features.position_payload(),
                    velocity=features.velocity_payload(),
                    metadata={
                        "source": self.metadata_source,
                        "model_id": loaded.model_id,
                        "model_score": score,
                        "threshold": settings.threshold,
                        "held_ms": held_ms,
                        "handedness": features.landmarks.handedness,
                        "model_group": group_for_static_pose_label(label),
                    },
                )
            )
        return events

    def reset(self) -> None:
        self._active_since.clear()
        self._cooldown_until.clear()

    def _ensure_settings(self) -> StaticPoseModelRuntimeSettings:
        if self._settings is None:
            self._settings = load_static_pose_runtime_settings()
        return self._settings

    def _ensure_loaded(self, settings: StaticPoseModelRuntimeSettings) -> LoadedStaticPoseModel:
        if self._loaded is None:
            self._loaded = load_static_pose_model(settings)
        return self._loaded

    def _predict_scores(self, features: GestureFeatures, loaded: LoadedStaticPoseModel) -> dict[str, float]:
        import torch

        landmarks = torch.tensor(
            [landmark_matrix(features, loaded.feature_spec)],
            dtype=torch.float32,
            device=loaded.device,
        )
        global_features = torch.tensor(
            [global_feature_vector(features, loaded.feature_spec)],
            dtype=torch.float32,
            device=loaded.device,
        )
        handedness = torch.tensor(
            [loaded.feature_spec.handedness_id(features.landmarks.handedness)],
            dtype=torch.long,
            device=loaded.device,
        )
        landmark_mask = torch.ones(
            1,
            loaded.feature_spec.num_landmarks,
            dtype=torch.bool,
            device=loaded.device,
        )
        with torch.inference_mode():
            outputs = loaded.model(
                {
                    "landmarks": landmarks,
                    "global_features": global_features,
                    "handedness": handedness,
                    "landmark_mask": landmark_mask,
                }
            )
            probabilities = torch.sigmoid(outputs["logits"])[0].detach().cpu().tolist()
        return {
            label: max(0.0, min(1.0, float(probabilities[index])))
            for index, label in enumerate(loaded.label_schema.labels)
        }


def load_static_pose_model(settings: StaticPoseModelRuntimeSettings) -> LoadedStaticPoseModel:
    try:
        import torch
        from hlgam.ml.models.static_pose_transformer import StaticPoseTransformer, StaticPoseTransformerConfig
    except ImportError as exc:
        raise RuntimeError("static pose model inference requires PyTorch; install the ml extra") from exc

    checkpoint_path = settings.checkpoint_path
    device = resolve_device(settings.device, torch)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"static pose checkpoint is missing model_state_dict: {checkpoint_path}")

    label_schema = LabelSchema.from_path(settings.label_schema_path)
    with settings.feature_schema_path.open("r", encoding="utf-8") as schema_file:
        feature_spec = LandmarkFeatureSpec.from_mapping(json.load(schema_file))
    training_config = checkpoint.get("config", {})
    model_section = training_config.get("model", {}) if isinstance(training_config, Mapping) else {}
    model_config = StaticPoseTransformerConfig(
        input_dim=feature_spec.per_landmark_dim,
        global_feature_dim=feature_spec.global_feature_dim,
        num_labels=len(label_schema.labels),
        d_model=int(model_section.get("d_model", 128)),
        num_layers=int(model_section.get("num_layers", 4)),
        num_heads=int(model_section.get("num_heads", 4)),
        ffn_dim=int(model_section.get("ffn_dim", 512)),
        dropout=float(model_section.get("dropout", 0.1)),
        use_global_features=bool(model_section.get("use_global_features", True)),
        auxiliary_heads=bool(model_section.get("auxiliary_heads", True)),
    )
    model = StaticPoseTransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    model_id = str(settings.model_id or checkpoint_path.parent.parent.name or checkpoint_path.stem)
    return LoadedStaticPoseModel(
        model=model,
        label_schema=label_schema,
        feature_spec=feature_spec,
        device=device,
        model_id=model_id,
        checkpoint_path=checkpoint_path,
    )


def load_static_pose_runtime_settings(env: Mapping[str, str] | None = None) -> StaticPoseModelRuntimeSettings:
    runtime_env = settings.static_pose_model_env()
    return StaticPoseModelRuntimeSettings(
        checkpoint_path=runtime_env.checkpoint_path,
        label_schema_path=runtime_env.label_schema_path,
        feature_schema_path=runtime_env.feature_schema_path,
        device=runtime_env.device,
        threshold=runtime_env.threshold,
        min_hold_ms=runtime_env.min_hold_ms,
        cooldown_ms=runtime_env.cooldown_ms,
        model_id=runtime_env.model_id,
    )


def group_for_static_pose_label(label: str) -> str | None:
    for group_name, group_labels in STATIC_POSE_MODEL_GROUP_LABELS.items():
        if label in group_labels:
            return group_name
    return None


def resolve_device(device_name: str, torch_module: Any) -> Any:
    if device_name == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(device_name)
