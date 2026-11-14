from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn

from hlgam.ml.models.components import FeedForwardHead, LandmarkTokenEncoder, SpatialHandEncoder


@dataclass(frozen=True)
class StaticPoseTransformerConfig:
    input_dim: int = 3
    global_feature_dim: int = 22
    num_labels: int = 28
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    handedness_classes: int = 3
    use_global_features: bool = True
    auxiliary_heads: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StaticPoseTransformer(nn.Module):
    def __init__(self, config: StaticPoseTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_encoder = LandmarkTokenEncoder(
            config.input_dim,
            config.d_model,
            handedness_classes=config.handedness_classes,
            dropout=config.dropout,
        )
        self.spatial_encoder = SpatialHandEncoder(
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            pooling="cls",
        )
        if config.use_global_features and config.global_feature_dim > 0:
            self.global_projection: nn.Module | None = nn.Sequential(
                nn.LayerNorm(config.global_feature_dim),
                nn.Linear(config.global_feature_dim, config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            head_input_dim = config.d_model * 2
        else:
            self.global_projection = None
            head_input_dim = config.d_model

        self.shared_projection = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.label_head = FeedForwardHead(config.d_model, config.num_labels, dropout=config.dropout)
        self.quality_head = FeedForwardHead(config.d_model, 1, dropout=config.dropout)
        if config.auxiliary_heads:
            self.finger_count_head: nn.Module | None = FeedForwardHead(config.d_model, 6, dropout=config.dropout)
            self.point_direction_head: nn.Module | None = FeedForwardHead(config.d_model, 9, dropout=config.dropout)
            self.flat_direction_head: nn.Module | None = FeedForwardHead(config.d_model, 5, dropout=config.dropout)
        else:
            self.finger_count_head = None
            self.point_direction_head = None
            self.flat_direction_head = None

    def forward(self, inputs: Mapping[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor]:
        if isinstance(inputs, Mapping):
            landmarks = inputs["landmarks"]
            handedness = inputs.get("handedness")
            landmark_mask = inputs.get("landmark_mask")
            global_features = inputs.get("global_features")
        else:
            landmarks = inputs
            handedness = None
            landmark_mask = None
            global_features = None

        tokens = self.token_encoder(landmarks, handedness)
        hand_embedding = self.spatial_encoder(tokens, landmark_mask)
        if self.global_projection is not None:
            if global_features is None:
                global_features = torch.zeros(
                    landmarks.shape[0],
                    self.config.global_feature_dim,
                    dtype=landmarks.dtype,
                    device=landmarks.device,
                )
            global_embedding = self.global_projection(global_features)
            embedding = torch.cat([hand_embedding, global_embedding], dim=-1)
        else:
            embedding = hand_embedding

        shared = self.shared_projection(embedding)
        outputs = {
            "logits": self.label_head(shared),
            "quality_logit": self.quality_head(shared).squeeze(-1),
            "embedding": shared,
        }
        if self.finger_count_head is not None:
            outputs["finger_count_logits"] = self.finger_count_head(shared)
        if self.point_direction_head is not None:
            outputs["point_direction_logits"] = self.point_direction_head(shared)
        if self.flat_direction_head is not None:
            outputs["flat_direction_logits"] = self.flat_direction_head(shared)
        return outputs
