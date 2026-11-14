from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from hlgam.models import GestureEvent, GestureFeatures


@dataclass(frozen=True)
class RecognizerConfig:
    enabled: bool = True


class GestureRecognizer(Protocol):
    name: str

    def process(self, features: GestureFeatures) -> list[GestureEvent]:
        ...

    def reset(self) -> None:
        ...


def clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))
