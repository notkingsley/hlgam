from __future__ import annotations

from dataclasses import dataclass, field

from hlgam.ml.inference.motion import MotionModelConfig
from hlgam.ml.inference.static_pose import StaticPoseModelConfig


@dataclass(frozen=True)
class RecognitionSwitchConfig:
    enabled: bool = True


@dataclass(frozen=True)
class LearnedModelsConfig:
    static_pose: StaticPoseModelConfig = field(default_factory=StaticPoseModelConfig)
    motion: MotionModelConfig = field(default_factory=MotionModelConfig)


@dataclass(frozen=True)
class RecognitionConfig:
    fsm: RecognitionSwitchConfig = field(default_factory=RecognitionSwitchConfig)
    models: LearnedModelsConfig = field(default_factory=LearnedModelsConfig)
