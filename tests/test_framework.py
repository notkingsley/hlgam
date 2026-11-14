from __future__ import annotations

from pathlib import Path

import pytest

from hlgam import GestureFramework
from hlgam.config import HLGAMConfig
from hlgam.gestures.engine import GestureEngine, GestureEngineConfig
from hlgam.gestures.pinch_drag import PinchDragConfig
from hlgam.ml.inference.motion import MotionModelConfig, MotionModelRuntimeSettings, MotionPrediction
from hlgam.ml.inference.static_pose import (
    StaticPoseModelConfig,
    StaticPoseModelGroupConfig,
    StaticPoseModelRuntimeSettings,
    load_static_pose_runtime_settings,
)
from hlgam.models import GestureEvent, GesturePhase, HandLandmarks
from hlgam.processing.feature_extractor import FeatureExtractor
from hlgam.processing.normalization import HandNormalizer
from hlgam.recognition import LearnedModelsConfig, RecognitionConfig, RecognitionSwitchConfig
from tests.helpers import synthetic_hand


def with_handedness(hand: HandLandmarks, handedness: str) -> HandLandmarks:
    return HandLandmarks(
        hand.landmarks,
        timestamp=hand.timestamp,
        handedness=handedness,
        score=hand.score,
        image_size=hand.image_size,
    )


def test_framework_processes_landmarks_and_invokes_callbacks() -> None:
    framework = GestureFramework()
    observed: list[str] = []
    framework.on_any(lambda event: observed.append(event.gesture))

    for timestamp, distance in [(0.0, 0.08), (0.13, 0.08), (0.20, 0.08), (0.30, 0.24)]:
        framework.process_landmarks(synthetic_hand(timestamp, distance))

    assert "PINCH_START" in observed
    assert "PINCH_HOLD" in observed
    assert "PINCH_END" in observed


def test_framework_processes_two_hand_gestures() -> None:
    framework = GestureFramework()

    framework.process_hands([
        with_handedness(synthetic_hand(0.0, offset_x=-0.20), "left"),
        with_handedness(synthetic_hand(0.0, offset_x=0.20), "right"),
    ])
    events = framework.process_hands([
        with_handedness(synthetic_hand(0.1, offset_x=-0.30), "left"),
        with_handedness(synthetic_hand(0.1, offset_x=0.30), "right"),
    ])

    assert "TWO_HAND_ZOOM_IN" in {event.gesture for event in events}


def test_framework_filters_disabled_gesture_events() -> None:
    framework = GestureFramework(HLGAMConfig(gesture_events={"PINCH_START": False}))
    observed: list[str] = []
    framework.on_any(lambda event: observed.append(event.gesture))

    for timestamp, distance in [(0.0, 0.08), (0.13, 0.08), (0.20, 0.08), (0.30, 0.24)]:
        framework.process_landmarks(synthetic_hand(timestamp, distance))

    assert "PINCH_START" not in observed
    assert "PINCH_HOLD" in observed
    assert "PINCH_END" in observed


def test_config_loads_gesture_event_toggles() -> None:
    config = HLGAMConfig.from_mapping({"gesture_events": {"OPEN_PALM": False, "WAVE": True}})

    assert config.gesture_events == {"OPEN_PALM": False, "WAVE": True}


def test_config_rejects_non_boolean_gesture_event_toggles() -> None:
    with pytest.raises(ValueError, match="gesture_events.OPEN_PALM must be a boolean"):
        HLGAMConfig.from_mapping({"gesture_events": {"OPEN_PALM": "false"}})


def test_config_loads_recognition_pipeline_toggles() -> None:
    config = HLGAMConfig.from_mapping(
        {
            "recognition": {
                "fsm": {"enabled": False},
                "models": {
                    "static_pose": {
                        "enabled": True,
                        "pose": {"enabled": False},
                        "finger_count": {"enabled": True},
                        "finger_pointing": {"enabled": False},
                        "flat_hand": {"enabled": False},
                    },
                    "motion": {
                        "enabled": True,
                    }
                },
            }
        }
    )

    assert config.recognition.fsm.enabled is False
    assert config.recognition.models.static_pose.enabled is True
    assert config.recognition.models.static_pose.pose.enabled is False
    assert config.recognition.models.static_pose.finger_count.enabled is True
    assert config.recognition.models.static_pose.finger_pointing.enabled is False
    assert config.recognition.models.static_pose.flat_hand.enabled is False
    assert config.recognition.models.motion.enabled is True


def test_config_rejects_static_model_artifact_fields() -> None:
    with pytest.raises(ValueError, match="unknown recognition.models.static_pose config parameter: checkpoint_path"):
        HLGAMConfig.from_mapping(
            {
                "recognition": {
                    "models": {
                        "static_pose": {
                            "enabled": True,
                            "checkpoint_path": "runs/static_pose/checkpoints/best.pt",
                        }
                    }
                }
            }
        )


def test_config_rejects_motion_model_artifact_fields() -> None:
    with pytest.raises(ValueError, match="unknown recognition.models.motion config parameter: checkpoint_path"):
        HLGAMConfig.from_mapping(
            {
                "recognition": {
                    "models": {
                        "motion": {
                            "enabled": True,
                            "checkpoint_path": "runs/motion/checkpoints/best.pt",
                        }
                    }
                }
            }
        )


def test_motion_runtime_settings_require_env_values() -> None:
    from hlgam.ml.inference.motion import load_motion_runtime_settings

    with pytest.raises(Exception, match="HLGAM_MOTION_MODEL_CHECKPOINT"):
        load_motion_runtime_settings({})


def test_static_pose_runtime_settings_require_env_values() -> None:
    with pytest.raises(RuntimeError, match="HLGAM_STATIC_POSE_MODEL_CHECKPOINT"):
        load_static_pose_runtime_settings({})


def test_learned_static_model_filters_fsm_matching_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    import hlgam.ml.inference.static_pose as static_pose_module

    monkeypatch.setattr(
        static_pose_module,
        "load_static_pose_runtime_settings",
        lambda: StaticPoseModelRuntimeSettings(
            checkpoint_path=Path("model.pt"),
            label_schema_path=Path("label_schema.json"),
            feature_schema_path=Path("input_feature_schema.json"),
            device="cpu",
            threshold=0.5,
            min_hold_ms=0,
            cooldown_ms=0,
            model_id="fake-static-model",
        ),
    )
    monkeypatch.setattr(
        static_pose_module,
        "load_static_pose_model",
        lambda settings: type("Loaded", (), {"model_id": "fake-static-model"})(),
    )
    monkeypatch.setattr(
        static_pose_module.LearnedStaticPoseRecognizer,
        "_predict_scores",
        lambda self, features, loaded: {
            "OPEN_PALM": 0.99,
            "FINGER_COUNT_5": 0.98,
            "FLAT_HAND_UP": 0.97,
        },
    )
    recognizer = static_pose_module.LearnedStaticPoseRecognizer(
        StaticPoseModelConfig(
            enabled=True,
            pose=StaticPoseModelGroupConfig(enabled=False),
            finger_count=StaticPoseModelGroupConfig(enabled=True),
            finger_pointing=StaticPoseModelGroupConfig(enabled=False),
            flat_hand=StaticPoseModelGroupConfig(enabled=False),
        )
    )

    events = recognizer.process(pinch_features())

    assert [event.gesture for event in events] == ["FINGER_COUNT_5"]
    assert events[0].metadata["model_group"] == "finger_count"


def test_learned_motion_model_buffers_sequence_and_emits_event(monkeypatch: pytest.MonkeyPatch) -> None:
    import hlgam.ml.inference.motion as motion_module

    monkeypatch.setattr(
        motion_module,
        "load_motion_runtime_settings",
        lambda: MotionModelRuntimeSettings(
            checkpoint_path=Path("model.pt"),
            label_schema_path=Path("label_schema.json"),
            feature_schema_path=Path("input_feature_schema.json"),
            device="cpu",
            threshold=0.5,
            min_sequence_frames=2,
            min_hold_ms=0,
            cooldown_ms=0,
            model_id="fake-motion-model",
        ),
    )
    monkeypatch.setattr(
        motion_module,
        "load_motion_model",
        lambda settings: type("Loaded", (), {"model_id": "fake-motion-model", "sequence_length": 2})(),
    )
    monkeypatch.setattr(
        motion_module.LearnedMotionRecognizer,
        "_predict",
        lambda self, features, loaded: [
            MotionPrediction(
                label="DRAG_MOVE",
                score=0.91,
                phase_label="active",
                phase_score=0.87,
                direction="right",
                quality_score=0.8,
                displacement=None,
                velocity=None,
                duration_ms=None,
                scale_delta=None,
                rotation_degrees=None,
                draw_point=None,
            )
        ],
    )
    recognizer = motion_module.LearnedMotionRecognizer(MotionModelConfig(enabled=True))

    assert recognizer.process(pinch_features(timestamp=0.0)) == []
    events = recognizer.process(pinch_features(timestamp=0.1))

    assert [event.gesture for event in events] == ["DRAG_MOVE"]
    assert events[0].phase == GesturePhase.UPDATED
    assert events[0].metadata["source"] == "learned_motion"
    assert events[0].metadata["model_id"] == "fake-motion-model"


def test_engine_can_disable_fsm_while_running_static_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import hlgam.gestures.engine as engine_module

    class FakeLearnedStaticPoseRecognizer:
        def __init__(self, config: StaticPoseModelConfig) -> None:
            self.config = config

        def process(self, features):
            if not self.config.enabled:
                return []
            return [
                GestureEvent(
                    gesture="OPEN_PALM",
                    phase=GesturePhase.COMPLETED,
                    confidence=0.99,
                    timestamp=features.timestamp,
                    metadata={"source": "fake_static_model"},
                )
            ]

        def reset(self) -> None:
            pass

    monkeypatch.setattr(engine_module, "LearnedStaticPoseRecognizer", FakeLearnedStaticPoseRecognizer)
    engine = GestureEngine(
        GestureEngineConfig(
            pinch_drag=PinchDragConfig(enabled=True, distance_threshold=0.10, release_threshold=0.16, hold_ms=0),
            recognition=RecognitionConfig(
                fsm=RecognitionSwitchConfig(enabled=False),
                models=LearnedModelsConfig(static_pose=StaticPoseModelConfig(enabled=True)),
            ),
        )
    )

    events = engine.process(pinch_features())

    assert [event.gesture for event in events] == ["OPEN_PALM"]


def test_engine_can_disable_fsm_while_running_motion_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import hlgam.gestures.engine as engine_module

    class FakeLearnedMotionRecognizer:
        def __init__(self, config: MotionModelConfig) -> None:
            self.config = config

        def process(self, features):
            if not self.config.enabled:
                return []
            return [
                GestureEvent(
                    gesture="SWIPE_RIGHT",
                    phase=GesturePhase.COMPLETED,
                    confidence=0.93,
                    timestamp=features.timestamp,
                    metadata={"source": "fake_motion_model"},
                )
            ]

        def reset(self) -> None:
            pass

    monkeypatch.setattr(engine_module, "LearnedMotionRecognizer", FakeLearnedMotionRecognizer)
    engine = GestureEngine(
        GestureEngineConfig(
            pinch_drag=PinchDragConfig(enabled=True, distance_threshold=0.10, release_threshold=0.16, hold_ms=0),
            recognition=RecognitionConfig(
                fsm=RecognitionSwitchConfig(enabled=False),
                models=LearnedModelsConfig(motion=MotionModelConfig(enabled=True)),
            ),
        )
    )

    events = engine.process(pinch_features())

    assert [event.gesture for event in events] == ["SWIPE_RIGHT"]


def test_engine_allows_fsm_and_static_model_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    import hlgam.gestures.engine as engine_module

    class FakeLearnedStaticPoseRecognizer:
        def __init__(self, config: StaticPoseModelConfig) -> None:
            self.config = config

        def process(self, features):
            return [
                GestureEvent(
                    gesture="OPEN_PALM",
                    phase=GesturePhase.COMPLETED,
                    confidence=0.99,
                    timestamp=features.timestamp,
                    metadata={"source": "fake_static_model"},
                )
            ] if self.config.enabled else []

        def reset(self) -> None:
            pass

    monkeypatch.setattr(engine_module, "LearnedStaticPoseRecognizer", FakeLearnedStaticPoseRecognizer)
    engine = GestureEngine(
        GestureEngineConfig(
            pinch_drag=PinchDragConfig(enabled=True, distance_threshold=0.10, release_threshold=0.16, hold_ms=0),
            recognition=RecognitionConfig(
                models=LearnedModelsConfig(static_pose=StaticPoseModelConfig(enabled=True)),
            ),
        )
    )

    gestures = [event.gesture for event in engine.process(pinch_features())]

    assert "PINCH_START" in gestures
    assert "OPEN_PALM" in gestures


def test_engine_allows_fsm_and_motion_model_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    import hlgam.gestures.engine as engine_module

    class FakeLearnedMotionRecognizer:
        def __init__(self, config: MotionModelConfig) -> None:
            self.config = config

        def process(self, features):
            return [
                GestureEvent(
                    gesture="SWIPE_RIGHT",
                    phase=GesturePhase.COMPLETED,
                    confidence=0.93,
                    timestamp=features.timestamp,
                    metadata={"source": "fake_motion_model"},
                )
            ] if self.config.enabled else []

        def reset(self) -> None:
            pass

    monkeypatch.setattr(engine_module, "LearnedMotionRecognizer", FakeLearnedMotionRecognizer)
    engine = GestureEngine(
        GestureEngineConfig(
            pinch_drag=PinchDragConfig(enabled=True, distance_threshold=0.10, release_threshold=0.16, hold_ms=0),
            recognition=RecognitionConfig(
                models=LearnedModelsConfig(motion=MotionModelConfig(enabled=True)),
            ),
        )
    )

    gestures = [event.gesture for event in engine.process(pinch_features())]

    assert "PINCH_START" in gestures
    assert "SWIPE_RIGHT" in gestures


def pinch_features(timestamp: float = 0.0):
    raw = synthetic_hand(timestamp=timestamp, normalized_pinch_distance=0.08)
    normalizer = HandNormalizer()
    extractor = FeatureExtractor()
    return extractor.extract(normalizer.normalize(raw), raw_landmarks=raw)
