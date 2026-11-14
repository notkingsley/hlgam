from __future__ import annotations

import torch
from torch import nn


class LandmarkTokenEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        *,
        num_landmarks: int = 21,
        handedness_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be greater than zero")
        if d_model <= 0:
            raise ValueError("d_model must be greater than zero")
        self.num_landmarks = num_landmarks
        self.feature_projection = nn.Linear(input_dim, d_model)
        self.landmark_embedding = nn.Embedding(num_landmarks, d_model)
        self.handedness_embedding = nn.Embedding(handedness_classes, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, landmarks: torch.Tensor, handedness: torch.Tensor | None = None) -> torch.Tensor:
        if landmarks.ndim != 3:
            raise ValueError("landmarks must have shape [batch, num_landmarks, feature_dim]")
        batch_size, num_landmarks, _ = landmarks.shape
        if num_landmarks != self.num_landmarks:
            raise ValueError(f"expected {self.num_landmarks} landmarks, received {num_landmarks}")

        device = landmarks.device
        landmark_ids = torch.arange(num_landmarks, device=device).unsqueeze(0).expand(batch_size, num_landmarks)
        if handedness is None:
            handedness = torch.zeros(batch_size, dtype=torch.long, device=device)
        handedness_tokens = self.handedness_embedding(handedness.to(device)).unsqueeze(1)
        tokens = self.feature_projection(landmarks)
        tokens = tokens + self.landmark_embedding(landmark_ids) + handedness_tokens
        return self.dropout(self.norm(tokens))


class SpatialHandEncoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be cls or mean")
        self.pooling = pooling
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if pooling == "cls" else None
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, landmark_mask: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch, num_tokens, d_model]")
        batch_size = tokens.shape[0]
        padding_mask = None
        if landmark_mask is not None:
            if landmark_mask.shape != tokens.shape[:2]:
                raise ValueError("landmark_mask must have shape [batch, num_tokens]")
            padding_mask = ~landmark_mask.bool()

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)
            if padding_mask is not None:
                cls_padding = torch.zeros(batch_size, 1, dtype=torch.bool, device=tokens.device)
                padding_mask = torch.cat([cls_padding, padding_mask], dim=1)

        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        if self.pooling == "cls":
            pooled = encoded[:, 0]
        else:
            if landmark_mask is None:
                pooled = encoded.mean(dim=1)
            else:
                weights = landmark_mask.float().unsqueeze(-1).clamp_min(0.0)
                pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.output_norm(pooled)


class TemporalSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        max_sequence_length: int = 64,
        dropout: float = 0.1,
        causal: bool = False,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be greater than zero")
        if d_model <= 0:
            raise ValueError("d_model must be greater than zero")
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be greater than zero")
        if pooling not in {"last", "cls", "mean"}:
            raise ValueError("pooling must be last, cls, or mean")
        self.max_sequence_length = max_sequence_length
        self.causal = causal
        self.pooling = pooling
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Embedding(max_sequence_length, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if pooling == "cls" else None
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        frame_embeddings: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if frame_embeddings.ndim != 3:
            raise ValueError("frame_embeddings must have shape [batch, time, input_dim]")
        batch_size, sequence_length, _ = frame_embeddings.shape
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds max_sequence_length={self.max_sequence_length}"
            )

        device = frame_embeddings.device
        positions = torch.arange(sequence_length, device=device).unsqueeze(0).expand(batch_size, sequence_length)
        tokens = self.input_projection(frame_embeddings) + self.position_embedding(positions)
        tokens = self.dropout(self.input_norm(tokens))
        padding_mask = None
        if frame_mask is not None:
            if frame_mask.shape != (batch_size, sequence_length):
                raise ValueError("frame_mask must have shape [batch, time]")
            padding_mask = ~frame_mask.bool()

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)
            if padding_mask is not None:
                cls_padding = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
                padding_mask = torch.cat([cls_padding, padding_mask], dim=1)

        causal_mask = None
        if self.causal:
            token_count = tokens.shape[1]
            causal_mask = torch.triu(torch.ones(token_count, token_count, dtype=torch.bool, device=device), diagonal=1)
            if self.cls_token is not None:
                causal_mask[0, :] = False
                causal_mask[:, 0] = False

        encoded = self.encoder(tokens, mask=causal_mask, src_key_padding_mask=padding_mask)
        if self.cls_token is not None:
            frame_tokens = encoded[:, 1:]
            pooled = encoded[:, 0]
        else:
            frame_tokens = encoded
            pooled = self._pool_frame_tokens(frame_tokens, frame_mask)
        return self.output_norm(pooled), self.output_norm(frame_tokens)

    def _pool_frame_tokens(self, frame_tokens: torch.Tensor, frame_mask: torch.Tensor | None) -> torch.Tensor:
        if self.pooling == "mean":
            if frame_mask is None:
                return frame_tokens.mean(dim=1)
            weights = frame_mask.float().unsqueeze(-1)
            return (frame_tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        if frame_mask is None:
            return frame_tokens[:, -1]
        valid_counts = frame_mask.long().sum(dim=1).clamp_min(1)
        last_indices = valid_counts - 1
        batch_indices = torch.arange(frame_tokens.shape[0], device=frame_tokens.device)
        return frame_tokens[batch_indices, last_indices]


class FeedForwardHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = hidden_dim or max(input_dim, output_dim * 2)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)
