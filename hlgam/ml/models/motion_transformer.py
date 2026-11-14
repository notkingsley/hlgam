from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn

from hlgam.ml.models.components import (
    FeedForwardHead,
    LandmarkTokenEncoder,
    SpatialHandEncoder,
    TemporalSequenceEncoder,
)


@dataclass(frozen=True)
class MotionGestureTransformerConfig:
    input_dim: int = 3
    global_feature_dim: int = 22
    num_labels: int = 40
    num_phases: int = 7
    spatial_d_model: int = 128
    spatial_layers: int = 2
    spatial_heads: int = 4
    spatial_ffn_dim: int = 512
    temporal_d_model: int = 256
    temporal_layers: int = 4
    temporal_heads: int = 8
    temporal_ffn_dim: int = 1024
    max_sequence_length: int = 32
    dropout: float = 0.1
    handedness_classes: int = 3
    use_global_features: bool = True
    causal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MotionGestureTransformer(nn.Module):
    def __init__(self, config: MotionGestureTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_encoder = LandmarkTokenEncoder(
            config.input_dim,
            config.spatial_d_model,
            handedness_classes=config.handedness_classes,
            dropout=config.dropout,
        )
        self.spatial_encoder = SpatialHandEncoder(
            d_model=config.spatial_d_model,
            num_layers=config.spatial_layers,
            num_heads=config.spatial_heads,
            ffn_dim=config.spatial_ffn_dim,
            dropout=config.dropout,
            pooling="cls",
        )
        if config.use_global_features and config.global_feature_dim > 0:
            self.global_projection: nn.Module | None = nn.Sequential(
                nn.LayerNorm(config.global_feature_dim),
                nn.Linear(config.global_feature_dim, config.spatial_d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            temporal_input_dim = config.spatial_d_model * 2
        else:
            self.global_projection = None
            temporal_input_dim = config.spatial_d_model

        self.temporal_encoder = TemporalSequenceEncoder(
            temporal_input_dim,
            d_model=config.temporal_d_model,
            num_layers=config.temporal_layers,
            num_heads=config.temporal_heads,
            ffn_dim=config.temporal_ffn_dim,
            max_sequence_length=config.max_sequence_length,
            dropout=config.dropout,
            causal=config.causal,
            pooling="last",
        )
        self.action_head = FeedForwardHead(config.temporal_d_model, config.num_labels, dropout=config.dropout)
        self.phase_head = FeedForwardHead(
            config.temporal_d_model,
            config.num_labels * config.num_phases,
            dropout=config.dropout,
        )
        self.direction_head = FeedForwardHead(config.temporal_d_model, 8, dropout=config.dropout)
        self.displacement_head = FeedForwardHead(config.temporal_d_model, 3, dropout=config.dropout)
        self.velocity_head = FeedForwardHead(config.temporal_d_model, 3, dropout=config.dropout)
        self.duration_head = FeedForwardHead(config.temporal_d_model, 1, dropout=config.dropout)
        self.scale_delta_head = FeedForwardHead(config.temporal_d_model, 1, dropout=config.dropout)
        self.rotation_head = FeedForwardHead(config.temporal_d_model, 1, dropout=config.dropout)
        self.draw_point_head = FeedForwardHead(config.temporal_d_model, 2, dropout=config.dropout)
        self.quality_head = FeedForwardHead(config.temporal_d_model, 1, dropout=config.dropout)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        landmarks = inputs["landmarks"]
        if landmarks.ndim != 4:
            raise ValueError("landmarks must have shape [batch, time, num_landmarks, feature_dim]")
        batch_size, sequence_length, num_landmarks, feature_dim = landmarks.shape
        frame_mask = inputs.get("frame_mask")
        landmark_mask = inputs.get("landmark_mask")
        handedness = inputs.get("handedness")
        global_features = inputs.get("global_features")

        flat_landmarks = landmarks.reshape(batch_size * sequence_length, num_landmarks, feature_dim)
        flat_handedness = flatten_handedness(handedness, batch_size, sequence_length, landmarks.device)
        if landmark_mask is None:
            flat_landmark_mask = None
        else:
            flat_landmark_mask = landmark_mask.reshape(batch_size * sequence_length, num_landmarks)

        tokens = self.token_encoder(flat_landmarks, flat_handedness)
        frame_embeddings = self.spatial_encoder(tokens, flat_landmark_mask).reshape(batch_size, sequence_length, -1)
        if self.global_projection is not None:
            if global_features is None:
                global_features = torch.zeros(
                    batch_size,
                    sequence_length,
                    self.config.global_feature_dim,
                    dtype=landmarks.dtype,
                    device=landmarks.device,
                )
            global_embeddings = self.global_projection(global_features)
            frame_embeddings = torch.cat([frame_embeddings, global_embeddings], dim=-1)

        embedding, temporal_tokens = self.temporal_encoder(frame_embeddings, frame_mask)
        phase_logits = self.phase_head(embedding).reshape(batch_size, self.config.num_labels, self.config.num_phases)
        return {
            "action_logits": self.action_head(embedding),
            "phase_logits": phase_logits,
            "direction_logits": self.direction_head(embedding),
            "displacement": self.displacement_head(embedding),
            "velocity": self.velocity_head(embedding),
            "duration_ms": self.duration_head(embedding).squeeze(-1),
            "scale_delta": self.scale_delta_head(embedding).squeeze(-1),
            "rotation_degrees": self.rotation_head(embedding).squeeze(-1),
            "draw_point": self.draw_point_head(embedding),
            "quality_logit": self.quality_head(embedding).squeeze(-1),
            "embedding": embedding,
            "frame_embeddings": temporal_tokens,
        }


def flatten_handedness(
    handedness: torch.Tensor | None,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor | None:
    if handedness is None:
        return None
    if handedness.ndim == 1:
        if handedness.shape[0] != batch_size:
            raise ValueError("handedness must have shape [batch] or [batch, time]")
        return handedness.to(device).unsqueeze(1).expand(batch_size, sequence_length).reshape(-1)
    if handedness.ndim == 2:
        if handedness.shape != (batch_size, sequence_length):
            raise ValueError("handedness must have shape [batch] or [batch, time]")
        return handedness.to(device).reshape(-1)
    raise ValueError("handedness must have shape [batch] or [batch, time]")
