from __future__ import annotations

import pytest

from hlgam.evaluation.metrics import ClassificationMetrics, LatencyTracker


def test_classification_metrics() -> None:
    metrics = ClassificationMetrics()
    metrics.add("PINCH", "PINCH")
    metrics.add("PINCH", "SWIPE")
    metrics.add("SWIPE", "SWIPE")

    assert metrics.accuracy() == pytest.approx(2 / 3)
    assert metrics.precision("SWIPE") == pytest.approx(0.5)
    assert metrics.recall("PINCH") == pytest.approx(0.5)


def test_latency_tracker() -> None:
    tracker = LatencyTracker()
    tracker.add(1.0, 1.1)
    tracker.add(2.0, 2.2)

    assert tracker.average_ms() == pytest.approx(150.0)
    assert tracker.percentile_ms(95) == pytest.approx(200.0)
