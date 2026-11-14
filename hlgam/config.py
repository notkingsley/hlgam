from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hlgam.gestures.air_draw import AirDrawConfig
from hlgam.gestures.flick import FlickConfig
from hlgam.gestures.motion import MotionGestureConfig
from hlgam.gestures.pinch_drag import PinchDragConfig
from hlgam.gestures.pose import FingerCountConfig, FingerPointingConfig, FlatHandConfig, StaticPoseConfig
from hlgam.gestures.rotation import RotationConfig
from hlgam.gestures.scroll import ScrollConfig
from hlgam.gestures.swipe import SwipeMotionConfig
from hlgam.gestures.two_hand import TwoHandGestureConfig
from hlgam.gestures.zoom import ZoomConfig
from hlgam.ml.inference.motion import MotionModelConfig
from hlgam.ml.inference.static_pose import StaticPoseModelConfig, StaticPoseModelGroupConfig
from hlgam.recognition import LearnedModelsConfig, RecognitionConfig, RecognitionSwitchConfig


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True


@dataclass(frozen=True)
class WebSocketConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    require_jwt: bool = True
    jwt_env_file: str = ".env"
    jwt_auth_timeout_seconds: float = 5.0


@dataclass(frozen=True)
class OSCConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 9000


@dataclass(frozen=True)
class BrokerConfig:
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    osc: OSCConfig = field(default_factory=OSCConfig)


@dataclass(frozen=True)
class GestureConfig:
    pinch_drag: PinchDragConfig = field(default_factory=PinchDragConfig)
    swipe_motion: SwipeMotionConfig = field(default_factory=SwipeMotionConfig)
    zoom: ZoomConfig = field(default_factory=ZoomConfig)
    pose: StaticPoseConfig = field(default_factory=StaticPoseConfig)
    finger_count: FingerCountConfig = field(default_factory=FingerCountConfig)
    finger_pointing: FingerPointingConfig = field(default_factory=FingerPointingConfig)
    flat_hand: FlatHandConfig = field(default_factory=FlatHandConfig)
    flick: FlickConfig = field(default_factory=FlickConfig)
    rotation: RotationConfig = field(default_factory=RotationConfig)
    scroll: ScrollConfig = field(default_factory=ScrollConfig)
    motion: MotionGestureConfig = field(default_factory=MotionGestureConfig)
    air_draw: AirDrawConfig = field(default_factory=AirDrawConfig)
    two_hand: TwoHandGestureConfig = field(default_factory=TwoHandGestureConfig)


@dataclass(frozen=True)
class HLGAMConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    gesture_events: dict[str, bool] = field(default_factory=dict)
    gesture_mapping: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> HLGAMConfig:
        camera = CameraConfig(**data.get("camera", {}))

        gesture_data = data.get("gestures", {})
        motion_data = _gesture_section(gesture_data, "motion")
        pinch_drag_data = _pinch_drag_gesture_section(gesture_data)
        swipe_motion_data = _combined_gesture_section(gesture_data, "swipe", "swipe_motion")
        flick_data = _gesture_section(gesture_data, "flick")
        rotation_data = _gesture_section(gesture_data, "rotation")
        scroll_data = _gesture_section(gesture_data, "scroll")
        air_draw_data = _gesture_section(gesture_data, "air_draw")
        _migrate_motion_subgroup_settings(
            motion_data, swipe_motion_data, flick_data, pinch_drag_data, rotation_data, scroll_data, air_draw_data
        )
        gestures = GestureConfig(
            pinch_drag=PinchDragConfig(**pinch_drag_data),
            swipe_motion=SwipeMotionConfig(**swipe_motion_data),
            zoom=ZoomConfig(**gesture_data.get("zoom", {})),
            pose=StaticPoseConfig(**gesture_data.get("pose", {})),
            finger_count=FingerCountConfig(**gesture_data.get("finger_count", {})),
            finger_pointing=FingerPointingConfig(**gesture_data.get("finger_pointing", {})),
            flat_hand=FlatHandConfig(**gesture_data.get("flat_hand", {})),
            flick=FlickConfig(**flick_data),
            rotation=RotationConfig(**rotation_data),
            scroll=ScrollConfig(**scroll_data),
            motion=MotionGestureConfig(**motion_data),
            air_draw=AirDrawConfig(**air_draw_data),
            two_hand=TwoHandGestureConfig(**gesture_data.get("two_hand", {})),
        )

        broker_data = data.get("broker", {})
        broker = BrokerConfig(
            websocket=WebSocketConfig(**broker_data.get("websocket", {})),
            osc=OSCConfig(**broker_data.get("osc", {})),
        )
        recognition = _recognition_config(data.get("recognition", {}))

        return cls(
            camera=camera,
            gestures=gestures,
            recognition=recognition,
            broker=broker,
            gesture_events=_bool_mapping(data.get("gesture_events", {}), "gesture_events"),
            gesture_mapping=dict(data.get("gesture_mapping", {})),
        )


def load_config(path: str | Path) -> HLGAMConfig:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return HLGAMConfig.from_mapping(json.loads(text))
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML config loading requires the PyYAML optional dependency") from exc
        loaded = yaml.safe_load(text) or {}
        return HLGAMConfig.from_mapping(loaded)
    raise ValueError(f"unsupported config file type: {config_path.suffix}")


def _bool_mapping(value: Any, name: str) -> dict[str, bool]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    result: dict[str, bool] = {}
    for key, enabled in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        if not isinstance(enabled, bool):
            raise ValueError(f"{name}.{key} must be a boolean")
        result[key] = enabled
    return result


def _recognition_config(value: Any) -> RecognitionConfig:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("recognition must be an object")
    fsm_data = value.get("fsm", {})
    if fsm_data is None:
        fsm_data = {}
    if not isinstance(fsm_data, dict):
        raise ValueError("recognition.fsm must be an object")
    models_data = value.get("models", {})
    if models_data is None:
        models_data = {}
    if not isinstance(models_data, dict):
        raise ValueError("recognition.models must be an object")
    return RecognitionConfig(
        fsm=RecognitionSwitchConfig(**fsm_data),
        models=LearnedModelsConfig(
            static_pose=_static_pose_model_config(models_data.get("static_pose", {})),
            motion=_motion_model_config(models_data.get("motion", {})),
        ),
    )


def _static_pose_model_config(value: Any) -> StaticPoseModelConfig:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("recognition.models.static_pose must be an object")
    allowed = {"enabled", "pose", "finger_count", "finger_pointing", "flat_hand"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown recognition.models.static_pose config parameter: {unknown[0]}")
    enabled = value.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("recognition.models.static_pose.enabled must be a boolean")
    return StaticPoseModelConfig(
        enabled=enabled,
        pose=_static_pose_model_group_config(value.get("pose", {}), "pose"),
        finger_count=_static_pose_model_group_config(value.get("finger_count", {}), "finger_count"),
        finger_pointing=_static_pose_model_group_config(value.get("finger_pointing", {}), "finger_pointing"),
        flat_hand=_static_pose_model_group_config(value.get("flat_hand", {}), "flat_hand"),
    )


def _static_pose_model_group_config(value: Any, name: str) -> StaticPoseModelGroupConfig:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"recognition.models.static_pose.{name} must be an object")
    unknown = sorted(set(value) - {"enabled"})
    if unknown:
        raise ValueError(f"unknown recognition.models.static_pose.{name} config parameter: {unknown[0]}")
    enabled = value.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"recognition.models.static_pose.{name}.enabled must be a boolean")
    return StaticPoseModelGroupConfig(enabled=enabled)


def _motion_model_config(value: Any) -> MotionModelConfig:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("recognition.models.motion must be an object")
    unknown = sorted(set(value) - {"enabled"})
    if unknown:
        raise ValueError(f"unknown recognition.models.motion config parameter: {unknown[0]}")
    enabled = value.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("recognition.models.motion.enabled must be a boolean")
    return MotionModelConfig(enabled=enabled)


def _combined_gesture_section(gesture_data: dict[str, Any], legacy_name: str, name: str) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    legacy_value = _gesture_section(gesture_data, legacy_name)
    value = _gesture_section(gesture_data, name)
    for section_name, section in ((legacy_name, legacy_value), (name, value)):
        if section is None:
            continue
        combined.update(section)
    return combined


def _gesture_section(gesture_data: dict[str, Any], name: str) -> dict[str, Any]:
    section = gesture_data.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"gestures.{name} must be an object")
    return dict(section)


def _pinch_drag_gesture_section(gesture_data: dict[str, Any]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    pinch_data = _gesture_section(gesture_data, "pinch")
    drag_data = _gesture_section(gesture_data, "drag")
    if "enabled" in pinch_data:
        combined["enabled"] = pinch_data["enabled"]
    elif "enabled" in drag_data:
        combined["enabled"] = drag_data["enabled"]
    for key in ("distance_threshold", "release_threshold", "hold_ms"):
        if key in pinch_data:
            combined[key] = pinch_data[key]

    legacy_drag_fields = {
        "pinch_threshold": "drag_pinch_threshold",
        "release_threshold": "drag_release_threshold",
        "activation_hold_ms": "drag_activation_hold_ms",
        "min_move": "drag_min_move",
    }
    for old_key, new_key in legacy_drag_fields.items():
        if old_key in drag_data:
            combined[new_key] = drag_data[old_key]

    combined.update(_gesture_section(gesture_data, "pinch_drag"))
    return combined


def _migrate_motion_subgroup_settings(
    motion_data: dict[str, Any],
    swipe_motion_data: dict[str, Any],
    flick_data: dict[str, Any],
    pinch_drag_data: dict[str, Any],
    rotation_data: dict[str, Any],
    scroll_data: dict[str, Any],
    air_draw_data: dict[str, Any],
) -> None:
    for key in ("diagonal_swipe_min_distance", "diagonal_swipe_axis_balance_ratio"):
        if key in motion_data:
            swipe_motion_data.setdefault(key, motion_data.pop(key))
    if "direction_ratio" in motion_data:
        swipe_motion_data.setdefault("direction_ratio", motion_data["direction_ratio"])
        flick_data.setdefault("direction_ratio", motion_data["direction_ratio"])
        scroll_data.setdefault("direction_ratio", motion_data["direction_ratio"])

    legacy_flick_fields = {
        "flick_min_velocity": "min_velocity",
        "flick_min_distance": "min_distance",
        "flick_max_duration_ms": "max_duration_ms",
    }
    for old_key, new_key in legacy_flick_fields.items():
        if old_key in motion_data:
            flick_data.setdefault(new_key, motion_data.pop(old_key))

    for key in (
        "pinch_tap_distance_threshold",
        "pinch_tap_release_threshold",
        "pinch_tap_min_duration_ms",
        "pinch_tap_max_duration_ms",
    ):
        if key in motion_data:
            pinch_drag_data.setdefault(key, motion_data.pop(key))
    if "air_draw_min_distance" in motion_data:
        air_draw_data.setdefault("min_distance", motion_data.pop("air_draw_min_distance"))

    for key in ("history_size", "max_history_ms", "cooldown_ms"):
        if key in motion_data:
            rotation_data.setdefault(key, motion_data[key])
            scroll_data.setdefault(key, motion_data[key])
    for key in (
        "circle_min_radius",
        "circle_min_arc_degrees",
        "circle_max_duration_ms",
        "circle_closure_ratio",
        "rotate_min_degrees",
        "rotate_max_duration_ms",
    ):
        if key in motion_data:
            rotation_data.setdefault(key, motion_data.pop(key))

    legacy_scroll_fields = {
        "scroll_min_distance": "min_distance",
        "scroll_max_duration_ms": "max_duration_ms",
    }
    for old_key, new_key in legacy_scroll_fields.items():
        if old_key in motion_data:
            scroll_data.setdefault(new_key, motion_data.pop(old_key))
