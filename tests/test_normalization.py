from __future__ import annotations

import pytest

from hlgam import landmark_indices as idx
from hlgam.processing.normalization import HandNormalizer
from tests.helpers import synthetic_hand


def test_normalization_anchors_wrist_and_scales_by_middle_mcp() -> None:
    raw = synthetic_hand(timestamp=0.0, normalized_pinch_distance=0.08, offset_x=0.4, offset_y=0.5)
    normalized = HandNormalizer().normalize(raw)

    assert normalized.point(idx.WRIST).x == pytest.approx(0.0)
    assert normalized.point(idx.WRIST).y == pytest.approx(0.0)
    assert normalized.point(idx.WRIST).z == pytest.approx(0.0)
    assert normalized.point(idx.WRIST).distance_to(normalized.point(idx.MIDDLE_MCP)) == pytest.approx(1.0)
    assert normalized.point(idx.THUMB_TIP).distance_to(normalized.point(idx.INDEX_TIP)) == pytest.approx(0.08)
