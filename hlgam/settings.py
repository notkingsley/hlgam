from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from hlgam import env

env.load("hlgam.env")


@dataclass(frozen=True)
class StaticPoseModelEnvSettings:
    checkpoint_path: Path
    label_schema_path: Path
    feature_schema_path: Path
    device: str
    threshold: float
    min_hold_ms: int
    cooldown_ms: int
    model_id: str | None = None


@dataclass(frozen=True)
class MotionModelEnvSettings:
    checkpoint_path: Path
    label_schema_path: Path
    feature_schema_path: Path
    device: str
    threshold: float
    min_sequence_frames: int
    min_hold_ms: int
    cooldown_ms: int
    model_id: str | None = None


@dataclass(frozen=True)
class JWTAuthEnvSettings:
    public_key: str | None = None
    public_key_file: Path | None = None
    audience: str | None = None
    issuer: str | None = None


HLGAM_STATIC_POSE_MODEL_CHECKPOINT = env.require("STATIC_POSE_MODEL_CHECKPOINT", Path)
HLGAM_STATIC_POSE_LABEL_SCHEMA = env.require("STATIC_POSE_LABEL_SCHEMA", Path)
HLGAM_STATIC_POSE_FEATURE_SCHEMA = env.require("STATIC_POSE_FEATURE_SCHEMA", Path)
HLGAM_STATIC_POSE_MODEL_DEVICE = env.require("STATIC_POSE_MODEL_DEVICE")
HLGAM_STATIC_POSE_MODEL_THRESHOLD = env.require("STATIC_POSE_MODEL_THRESHOLD", float)
HLGAM_STATIC_POSE_MODEL_MIN_HOLD_MS = env.require("STATIC_POSE_MODEL_MIN_HOLD_MS", int)
HLGAM_STATIC_POSE_MODEL_COOLDOWN_MS = env.require("STATIC_POSE_MODEL_COOLDOWN_MS", int)
HLGAM_STATIC_POSE_MODEL_ID = env.optional("STATIC_POSE_MODEL_ID", None)

HLGAM_WS_JWT_PUBLIC_KEY = env.optional("HLGAM_WS_JWT_PUBLIC_KEY", None)
HLGAM_WS_JWT_PUBLIC_KEY_FILE = env.optional("HLGAM_WS_JWT_PUBLIC_KEY_FILE", None)
HLGAM_WS_JWT_AUDIENCE = env.optional("HLGAM_WS_JWT_AUDIENCE", None)
HLGAM_WS_JWT_ISSUER = env.optional("HLGAM_WS_JWT_ISSUER", None)


STATIC_POSE_ENV_SETTINGS = StaticPoseModelEnvSettings(
    checkpoint_path=HLGAM_STATIC_POSE_MODEL_CHECKPOINT,
    label_schema_path=HLGAM_STATIC_POSE_LABEL_SCHEMA,
    feature_schema_path=HLGAM_STATIC_POSE_FEATURE_SCHEMA,
    device=HLGAM_STATIC_POSE_MODEL_DEVICE,
    threshold=HLGAM_STATIC_POSE_MODEL_THRESHOLD,
    min_hold_ms=HLGAM_STATIC_POSE_MODEL_MIN_HOLD_MS,
    cooldown_ms=HLGAM_STATIC_POSE_MODEL_COOLDOWN_MS,
    model_id=HLGAM_STATIC_POSE_MODEL_ID,
)


def static_pose_model_env():
    return STATIC_POSE_ENV_SETTINGS


def motion_model_env(values: Mapping[str, str] | None = None) -> MotionModelEnvSettings:
    env_values = env.environ_values() if values is None else values
    return MotionModelEnvSettings(
        checkpoint_path=env.require_from(env_values, "HLGAM_MOTION_MODEL_CHECKPOINT", Path),
        label_schema_path=env.require_from(env_values, "HLGAM_MOTION_LABEL_SCHEMA", Path),
        feature_schema_path=env.require_from(env_values, "HLGAM_MOTION_FEATURE_SCHEMA", Path),
        device=env.optional_from(env_values, "HLGAM_MOTION_MODEL_DEVICE", "auto"),
        threshold=env.optional_from(env_values, "HLGAM_MOTION_MODEL_THRESHOLD", 0.5, float),
        min_sequence_frames=env.optional_from(env_values, "HLGAM_MOTION_MODEL_MIN_SEQUENCE_FRAMES", 4, int),
        min_hold_ms=env.optional_from(env_values, "HLGAM_MOTION_MODEL_MIN_HOLD_MS", 0, int),
        cooldown_ms=env.optional_from(env_values, "HLGAM_MOTION_MODEL_COOLDOWN_MS", 250, int),
        model_id=env.optional_from(env_values, "HLGAM_MOTION_MODEL_ID", None),
    )


def jwt_auth_env() -> JWTAuthEnvSettings:
    return JWTAuthEnvSettings(
        public_key=HLGAM_WS_JWT_PUBLIC_KEY,
        public_key_file=HLGAM_WS_JWT_PUBLIC_KEY_FILE,
        audience=HLGAM_WS_JWT_AUDIENCE,
        issuer=HLGAM_WS_JWT_ISSUER,
    )
