from __future__ import annotations

import pytest

from hlgam.processing.feature_extractor import FeatureExtractor
from hlgam.processing.normalization import HandNormalizer
from tests.helpers import synthetic_hand


def test_feature_extractor_uses_normalized_shape_and_raw_motion() -> None:
    normalizer = HandNormalizer()
    extractor = FeatureExtractor()

    first_raw = synthetic_hand(timestamp=0.0, normalized_pinch_distance=0.08, offset_x=0.0)
    first = extractor.extract(normalizer.normalize(first_raw), raw_landmarks=first_raw)

    second_raw = synthetic_hand(timestamp=0.1, normalized_pinch_distance=0.08, offset_x=0.05)
    second = extractor.extract(normalizer.normalize(second_raw), raw_landmarks=second_raw)

    assert first.pinch_distance == pytest.approx(0.08)
    assert second.pinch_distance == pytest.approx(0.08)
    assert second.velocity.x == pytest.approx(0.5)
    assert second.speed == pytest.approx(0.5)
    assert second.direction.x == pytest.approx(1.0)


def test_feature_extractor_reports_finger_curl_states() -> None:
    normalizer = HandNormalizer()
    extractor = FeatureExtractor()
    raw = synthetic_hand(timestamp=0.0)

    features = extractor.extract(normalizer.normalize(raw), raw_landmarks=raw)

    assert set(features.finger_curl) == {"thumb", "index", "middle", "ring", "pinky"}
    assert features.finger_curl["middle"] is False
