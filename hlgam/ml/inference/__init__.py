"""Runtime inference adapters for learned HLGAM models."""

from hlgam.ml.inference.motion import (
    LearnedMotionRecognizer,
    MotionModelConfig,
    MotionModelRuntimeSettings,
)
from hlgam.ml.inference.static_pose import (
    LearnedStaticPoseRecognizer,
    StaticPoseModelConfig,
    StaticPoseModelGroupConfig,
    StaticPoseModelRuntimeSettings,
)

__all__ = [
    "LearnedMotionRecognizer",
    "LearnedStaticPoseRecognizer",
    "MotionModelConfig",
    "MotionModelRuntimeSettings",
    "StaticPoseModelConfig",
    "StaticPoseModelGroupConfig",
    "StaticPoseModelRuntimeSettings",
]
