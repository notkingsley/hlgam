from __future__ import annotations

from typing import Mapping, Sequence

import torch


def multilabel_classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    label_names: Sequence[str] | None = None,
    threshold: float | torch.Tensor = 0.5,
) -> dict[str, object]:
    probabilities = torch.sigmoid(logits.detach()).cpu()
    targets = targets.detach().cpu().float()
    if isinstance(threshold, torch.Tensor):
        threshold_tensor = threshold.detach().cpu().float()
    else:
        threshold_tensor = torch.full((probabilities.shape[1],), float(threshold))
    predictions = (probabilities >= threshold_tensor).float()

    true_positive = (predictions * targets).sum(dim=0)
    false_positive = (predictions * (1 - targets)).sum(dim=0)
    false_negative = ((1 - predictions) * targets).sum(dim=0)
    true_negative = ((1 - predictions) * (1 - targets)).sum(dim=0)

    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    micro_precision = safe_divide(true_positive.sum(), true_positive.sum() + false_positive.sum())
    micro_recall = safe_divide(true_positive.sum(), true_positive.sum() + false_negative.sum())
    micro_f1 = safe_divide(2 * micro_precision * micro_recall, micro_precision + micro_recall)
    exact_match = (predictions == targets).all(dim=1).float().mean() if targets.numel() else torch.tensor(0.0)
    hamming_loss = (predictions != targets).float().mean() if targets.numel() else torch.tensor(0.0)

    names = list(label_names or [str(index) for index in range(probabilities.shape[1])])
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(targets[:, index].sum().item()),
            "predicted": int(predictions[:, index].sum().item()),
            "true_positive": int(true_positive[index].item()),
            "false_positive": int(false_positive[index].item()),
            "false_negative": int(false_negative[index].item()),
            "true_negative": int(true_negative[index].item()),
        }
        for index, label in enumerate(names)
    }

    return {
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "exact_match": float(exact_match),
        "hamming_loss": float(hamming_loss),
        "per_class": per_class,
    }


def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1e-12), torch.zeros_like(denominator))


def flatten_scalar_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if isinstance(value, int | float)}
