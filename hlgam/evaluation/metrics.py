from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class ClassificationMetrics:
    true_labels: list[str] = field(default_factory=list)
    predicted_labels: list[str] = field(default_factory=list)

    def add(self, true_label: str, predicted_label: str) -> None:
        self.true_labels.append(true_label)
        self.predicted_labels.append(predicted_label)

    def accuracy(self) -> float:
        if not self.true_labels:
            return 0.0
        pairs = zip(self.true_labels, self.predicted_labels, strict=True)
        correct = sum(true == predicted for true, predicted in pairs)
        return correct / len(self.true_labels)

    def precision(self, label: str) -> float:
        pairs = zip(self.true_labels, self.predicted_labels, strict=True)
        true_positive = sum(true == predicted == label for true, predicted in pairs)
        predicted_positive = Counter(self.predicted_labels)[label]
        if predicted_positive == 0:
            return 0.0
        return true_positive / predicted_positive

    def recall(self, label: str) -> float:
        pairs = zip(self.true_labels, self.predicted_labels, strict=True)
        true_positive = sum(true == predicted == label for true, predicted in pairs)
        actual_positive = Counter(self.true_labels)[label]
        if actual_positive == 0:
            return 0.0
        return true_positive / actual_positive

    def f1(self, label: str) -> float:
        precision = self.precision(label)
        recall = self.recall(label)
        if precision + recall == 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)


@dataclass
class LatencyTracker:
    samples_ms: list[float] = field(default_factory=list)

    def add(self, started_at: float, ended_at: float) -> None:
        self.samples_ms.append(max(0.0, (ended_at - started_at) * 1000.0))

    def average_ms(self) -> float:
        if not self.samples_ms:
            return 0.0
        return sum(self.samples_ms) / len(self.samples_ms)

    def percentile_ms(self, percentile: float) -> float:
        if not self.samples_ms:
            return 0.0
        ordered = sorted(self.samples_ms)
        index = min(len(ordered) - 1, max(0, round((percentile / 100.0) * (len(ordered) - 1))))
        return ordered[index]
