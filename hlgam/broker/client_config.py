from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass, replace
from typing import Any

from hlgam.gestures.engine import GestureEngineConfig


class ClientConfigError(ValueError):
    pass


NON_GESTURE_CONFIG_FIELDS = {"gesture_mapping", "gesture_events", "recognition"}
GESTURE_NAMES = tuple(
    field.name for field in fields(GestureEngineConfig) if field.name not in NON_GESTURE_CONFIG_FIELDS
)
CONFIG_COMMANDS = {"config", "configure", "set_config", "update_config"}


def is_config_message(payload: dict[str, Any]) -> bool:
    command = payload.get("type", payload.get("command"))
    if isinstance(command, str) and command.casefold() in CONFIG_COMMANDS:
        return True
    return any(
        key in payload
        for key in (
            "config",
            "gestures",
            "recognition",
            "models",
            "gesture_mapping",
            "mapping",
            "gesture_events",
            "enabled_gestures",
        )
    )


def merge_client_engine_config(base: GestureEngineConfig, payload: dict[str, Any]) -> GestureEngineConfig:
    config_payload = payload.get("config", {})
    if config_payload is None:
        config_payload = {}
    if not isinstance(config_payload, dict):
        raise ClientConfigError("config must be an object")

    gestures_payload = _combined_object(payload.get("gestures"), config_payload.get("gestures"), "gestures")
    mapping_payload = _combined_object(
        payload.get("gesture_mapping", payload.get("mapping")),
        config_payload.get("gesture_mapping", config_payload.get("mapping")),
        "gesture_mapping",
    )
    events_payload = _combined_object(
        payload.get("gesture_events", payload.get("enabled_gestures")),
        config_payload.get("gesture_events", config_payload.get("enabled_gestures")),
        "gesture_events",
    )
    recognition_payload = _combined_object(
        payload.get("recognition"),
        config_payload.get("recognition"),
        "recognition",
    )
    models_payload = _combined_object(payload.get("models"), config_payload.get("models"), "models")
    if models_payload:
        recognition_payload = dict(recognition_payload)
        recognition_models = _combined_object(recognition_payload.get("models"), models_payload, "recognition.models")
        recognition_payload["models"] = recognition_models
    gestures_payload = _migrate_legacy_gesture_payload(gestures_payload)

    gesture_configs = {gesture_name: getattr(base, gesture_name) for gesture_name in GESTURE_NAMES}
    for gesture_name, patch in gestures_payload.items():
        if gesture_name not in gesture_configs:
            raise ClientConfigError(f"unknown gesture config: {gesture_name}")
        if not isinstance(patch, dict):
            raise ClientConfigError(f"gesture config for {gesture_name} must be an object")
        gesture_configs[gesture_name] = merge_dataclass_config(gesture_configs[gesture_name], patch, gesture_name)

    gesture_mapping = dict(base.gesture_mapping)
    for gesture, command in mapping_payload.items():
        if not isinstance(gesture, str) or not gesture:
            raise ClientConfigError("gesture_mapping keys must be non-empty strings")
        if command is None:
            gesture_mapping.pop(gesture, None)
        elif isinstance(command, str) and command:
            gesture_mapping[gesture] = command
        else:
            raise ClientConfigError(f"gesture_mapping value for {gesture} must be a non-empty string or null")

    gesture_events = dict(base.gesture_events)
    for gesture, enabled in events_payload.items():
        if not isinstance(gesture, str) or not gesture:
            raise ClientConfigError("gesture_events keys must be non-empty strings")
        if not isinstance(enabled, bool):
            raise ClientConfigError(f"gesture_events value for {gesture} must be a boolean")
        gesture_events[gesture] = enabled

    recognition = base.recognition
    if recognition_payload:
        recognition = merge_dataclass_config(base.recognition, recognition_payload, "recognition")

    return replace(
        base,
        **gesture_configs,
        recognition=recognition,
        gesture_events=gesture_events,
        gesture_mapping=gesture_mapping,
    )


def merge_dataclass_config(config: Any, patch: dict[str, Any], namespace: str) -> Any:
    if not is_dataclass(config):
        raise ClientConfigError(f"{namespace} config is not a dataclass")

    allowed = {field.name: getattr(config, field.name) for field in fields(config)}
    updates: dict[str, Any] = {}
    for key, value in patch.items():
        if key not in allowed:
            raise ClientConfigError(f"unknown {namespace} config parameter: {key}")
        updates[key] = _coerce_like(allowed[key], value, f"{namespace}.{key}")
    return replace(config, **updates)


def _combined_object(primary: Any, secondary: Any, name: str) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for value in (secondary, primary):
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ClientConfigError(f"{name} must be an object")
        combined.update(value)
    return combined


def _migrate_legacy_gesture_payload(gestures_payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(gestures_payload)
    pinch_drag_patch = _gesture_patch_object(migrated.get("pinch_drag"), "pinch_drag")
    rotation_patch = _gesture_patch_object(migrated.get("rotation"), "rotation")
    scroll_patch = _gesture_patch_object(migrated.get("scroll"), "scroll")
    air_draw_patch = _gesture_patch_object(migrated.get("air_draw"), "air_draw")

    pinch_patch = migrated.pop("pinch", None)
    if pinch_patch is not None:
        pinch_patch = _gesture_patch_object(pinch_patch, "pinch")
        if "enabled" in pinch_patch:
            pinch_drag_patch.setdefault("enabled", pinch_patch["enabled"])
        for key in ("distance_threshold", "release_threshold", "hold_ms"):
            if key in pinch_patch:
                pinch_drag_patch.setdefault(key, pinch_patch[key])

    drag_patch = migrated.pop("drag", None)
    if drag_patch is not None:
        drag_patch = _gesture_patch_object(drag_patch, "drag")
        if "enabled" in drag_patch:
            pinch_drag_patch.setdefault("enabled", drag_patch["enabled"])
        legacy_drag_fields = {
            "pinch_threshold": "drag_pinch_threshold",
            "release_threshold": "drag_release_threshold",
            "activation_hold_ms": "drag_activation_hold_ms",
            "min_move": "drag_min_move",
        }
        for old_key, new_key in legacy_drag_fields.items():
            if old_key in drag_patch:
                pinch_drag_patch.setdefault(new_key, drag_patch[old_key])

    motion_patch = migrated.get("motion")
    if motion_patch is not None:
        motion_patch = _gesture_patch_object(motion_patch, "motion")
        migrated["motion"] = motion_patch
        for key in (
            "pinch_tap_distance_threshold",
            "pinch_tap_release_threshold",
            "pinch_tap_min_duration_ms",
            "pinch_tap_max_duration_ms",
        ):
            if key in motion_patch:
                pinch_drag_patch.setdefault(key, motion_patch.pop(key))
        if "air_draw_min_distance" in motion_patch:
            air_draw_patch.setdefault("min_distance", motion_patch.pop("air_draw_min_distance"))
        for key in ("history_size", "max_history_ms", "cooldown_ms"):
            if key in motion_patch:
                rotation_patch.setdefault(key, motion_patch[key])
                scroll_patch.setdefault(key, motion_patch[key])
        if "direction_ratio" in motion_patch:
            scroll_patch.setdefault("direction_ratio", motion_patch["direction_ratio"])
        for key in (
            "circle_min_radius",
            "circle_min_arc_degrees",
            "circle_max_duration_ms",
            "circle_closure_ratio",
            "rotate_min_degrees",
            "rotate_max_duration_ms",
        ):
            if key in motion_patch:
                rotation_patch.setdefault(key, motion_patch.pop(key))
        legacy_scroll_fields = {
            "scroll_min_distance": "min_distance",
            "scroll_max_duration_ms": "max_duration_ms",
        }
        for old_key, new_key in legacy_scroll_fields.items():
            if old_key in motion_patch:
                scroll_patch.setdefault(new_key, motion_patch.pop(old_key))

    if pinch_drag_patch:
        migrated["pinch_drag"] = pinch_drag_patch
    if rotation_patch:
        migrated["rotation"] = rotation_patch
    if scroll_patch:
        migrated["scroll"] = scroll_patch
    if air_draw_patch:
        migrated["air_draw"] = air_draw_patch
    return migrated


def _gesture_patch_object(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ClientConfigError(f"gesture config for {name} must be an object")
    return dict(value)


def _coerce_like(current_value: Any, value: Any, name: str) -> Any:
    if is_dataclass(current_value):
        if not isinstance(value, dict):
            raise ClientConfigError(f"{name} must be an object")
        return merge_dataclass_config(current_value, value, name)
    if isinstance(current_value, bool):
        if not isinstance(value, bool):
            raise ClientConfigError(f"{name} must be a boolean")
        return value
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ClientConfigError(f"{name} must be an integer")
        if value < 0:
            raise ClientConfigError(f"{name} must be greater than or equal to 0")
        return value
    if isinstance(current_value, float):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ClientConfigError(f"{name} must be a number")
        if value < 0:
            raise ClientConfigError(f"{name} must be greater than or equal to 0")
        return float(value)
    if isinstance(current_value, str):
        if not isinstance(value, str):
            raise ClientConfigError(f"{name} must be a string")
        return value
    raise ClientConfigError(f"{name} has unsupported config value type")


def config_payload(config: GestureEngineConfig) -> dict[str, Any]:
    return {
        "recognition": asdict(config.recognition),
        "gesture_events": config.gesture_events,
        "gesture_mapping": config.gesture_mapping,
    }
