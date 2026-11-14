from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class StaticPoseLossConfig:
    primary_loss: str = "bce"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    finger_count_weight: float = 0.2
    point_direction_weight: float = 0.2
    flat_direction_weight: float = 0.2


@dataclass(frozen=True)
class MotionGestureLossConfig:
    primary_loss: str = "bce"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    phase_weight: float = 0.5
    direction_weight: float = 0.1
    displacement_weight: float = 0.1
    velocity_weight: float = 0.1
    duration_weight: float = 0.05
    scale_weight: float = 0.05
    rotation_weight: float = 0.05
    draw_weight: float = 0.05


class StaticPoseLoss(nn.Module):
    def __init__(self, config: StaticPoseLossConfig | None = None, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.config = config or StaticPoseLossConfig()
        if self.config.primary_loss not in {"bce", "focal"}:
            raise ValueError("primary_loss must be bce or focal")
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits = outputs["logits"]
        targets = batch["labels"].to(logits.device)
        if self.config.primary_loss == "focal":
            primary_loss = multilabel_focal_loss(
                logits,
                targets,
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma,
                pos_weight=self.pos_weight,
            )
        else:
            primary_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)

        loss = primary_loss
        parts = {"primary": float(primary_loss.detach().cpu())}
        aux = batch.get("aux", {})
        loss = loss + self._add_aux_loss(
            outputs,
            aux,
            "finger_count_logits",
            "finger_count",
            self.config.finger_count_weight,
            parts,
        )
        loss = loss + self._add_aux_loss(
            outputs,
            aux,
            "point_direction_logits",
            "point_direction",
            self.config.point_direction_weight,
            parts,
        )
        loss = loss + self._add_aux_loss(
            outputs,
            aux,
            "flat_direction_logits",
            "flat_direction",
            self.config.flat_direction_weight,
            parts,
        )
        parts["total"] = float(loss.detach().cpu())
        return loss, parts

    def _add_aux_loss(
        self,
        outputs: Mapping[str, torch.Tensor],
        aux: Mapping[str, torch.Tensor],
        logits_key: str,
        target_key: str,
        weight: float,
        parts: dict[str, float],
    ) -> torch.Tensor:
        if weight <= 0 or logits_key not in outputs or target_key not in aux:
            return torch.zeros((), device=outputs["logits"].device)
        logits = outputs[logits_key]
        targets = aux[target_key].to(logits.device)
        if not torch.any(targets != -100):
            return torch.zeros((), device=logits.device)
        aux_loss = F.cross_entropy(logits, targets, ignore_index=-100)
        parts[target_key] = float(aux_loss.detach().cpu())
        return aux_loss * weight


class MotionGestureLoss(nn.Module):
    def __init__(self, config: MotionGestureLossConfig | None = None, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.config = config or MotionGestureLossConfig()
        if self.config.primary_loss not in {"bce", "focal"}:
            raise ValueError("primary_loss must be bce or focal")
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits = outputs["action_logits"]
        targets = batch["labels"].to(logits.device)
        if self.config.primary_loss == "focal":
            action_loss = multilabel_focal_loss(
                logits,
                targets,
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma,
                pos_weight=self.pos_weight,
            )
        else:
            action_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)

        loss = action_loss
        parts = {"action": float(action_loss.detach().cpu())}
        loss = loss + self._phase_loss(outputs, batch, parts)
        aux = batch.get("aux", {})
        loss = loss + self._direction_loss(outputs, aux, parts)
        loss = loss + self._regression_loss(outputs, aux, "displacement", self.config.displacement_weight, parts)
        loss = loss + self._regression_loss(outputs, aux, "velocity", self.config.velocity_weight, parts)
        loss = loss + self._regression_loss(outputs, aux, "duration_ms", self.config.duration_weight, parts)
        loss = loss + self._regression_loss(outputs, aux, "scale_delta", self.config.scale_weight, parts)
        loss = loss + self._regression_loss(outputs, aux, "rotation_degrees", self.config.rotation_weight, parts)
        loss = loss + self._regression_loss(outputs, aux, "draw_point", self.config.draw_weight, parts)
        parts["total"] = float(loss.detach().cpu())
        return loss, parts

    def _phase_loss(
        self,
        outputs: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
        parts: dict[str, float],
    ) -> torch.Tensor:
        if self.config.phase_weight <= 0 or "phase_logits" not in outputs:
            return torch.zeros((), device=outputs["action_logits"].device)
        phase_logits = outputs["phase_logits"]
        phase_targets = batch["phase_targets"].to(phase_logits.device)
        phase_mask = batch["phase_mask"].to(phase_logits.device).bool()
        if not torch.any(phase_mask):
            return torch.zeros((), device=phase_logits.device)
        phase_loss = F.cross_entropy(phase_logits[phase_mask], phase_targets[phase_mask])
        parts["phase"] = float(phase_loss.detach().cpu())
        return phase_loss * self.config.phase_weight

    def _direction_loss(
        self,
        outputs: Mapping[str, torch.Tensor],
        aux: Mapping[str, torch.Tensor],
        parts: dict[str, float],
    ) -> torch.Tensor:
        if self.config.direction_weight <= 0 or "direction_logits" not in outputs or "direction" not in aux:
            return torch.zeros((), device=outputs["action_logits"].device)
        logits = outputs["direction_logits"]
        targets = aux["direction"].to(logits.device)
        mask = aux.get("direction_mask", torch.ones_like(targets, dtype=torch.bool)).to(logits.device).bool()
        if not torch.any(mask):
            return torch.zeros((), device=logits.device)
        direction_loss = F.cross_entropy(logits[mask], targets[mask])
        parts["direction"] = float(direction_loss.detach().cpu())
        return direction_loss * self.config.direction_weight

    def _regression_loss(
        self,
        outputs: Mapping[str, torch.Tensor],
        aux: Mapping[str, torch.Tensor],
        target_name: str,
        weight: float,
        parts: dict[str, float],
    ) -> torch.Tensor:
        if weight <= 0 or target_name not in outputs or target_name not in aux:
            return torch.zeros((), device=outputs["action_logits"].device)
        predictions = outputs[target_name]
        targets = aux[target_name].to(predictions.device)
        mask_name = regression_mask_name(target_name)
        mask = aux.get(mask_name)
        if mask is None:
            mask = torch.ones(targets.shape[0], dtype=torch.bool, device=predictions.device)
        else:
            mask = mask.to(predictions.device).bool()
        if not torch.any(mask):
            return torch.zeros((), device=predictions.device)
        regression_loss = F.smooth_l1_loss(predictions[mask], targets[mask])
        parts[target_name] = float(regression_loss.detach().cpu())
        return regression_loss * weight


def regression_mask_name(target_name: str) -> str:
    return {
        "duration_ms": "duration_mask",
        "scale_delta": "scale_mask",
        "rotation_degrees": "rotation_mask",
        "draw_point": "draw_mask",
    }.get(target_name, f"{target_name}_mask")


def multilabel_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    probabilities = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probabilities, 1 - probabilities)
    alpha_factor = torch.where(targets == 1, alpha, 1 - alpha)
    loss = alpha_factor * (1 - pt).pow(gamma) * bce
    return loss.mean()
