from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hlgam import settings as app_settings


class JWTAuthError(RuntimeError):
    pass


@dataclass(frozen=True)
class JWTAuthSettings:
    algorithms: tuple[str, ...] = ("EdDSA",)
    required_claims: tuple[str, ...] = ("exp",)
    leeway_seconds: int = 0


@dataclass(frozen=True)
class JWTVerificationContext:
    public_key: str
    audience: str | None = None
    issuer: str | None = None
    algorithms: tuple[str, ...] = field(default_factory=lambda: ("EdDSA",))
    required_claims: tuple[str, ...] = field(default_factory=lambda: ("exp",))
    leeway_seconds: int = 0


class JWTVerifier:
    def __init__(self, settings: JWTAuthSettings | None = None) -> None:
        self.settings = settings or JWTAuthSettings()
        self._context: JWTVerificationContext | None = None

    def decode(self, token: str) -> dict[str, Any]:
        try:
            import jwt
        except ImportError as exc:
            raise JWTAuthError(
                "JWT authentication requires the PyJWT[crypto] dependency"
            ) from exc

        context = self._get_context()
        try:
            claims = jwt.decode(
                token,
                context.public_key,
                algorithms=list(context.algorithms),
                audience=context.audience,
                issuer=context.issuer,
                leeway=context.leeway_seconds,
                options={
                    "require": list(context.required_claims),
                    "verify_aud": context.audience is not None,
                    "verify_iss": context.issuer is not None,
                },
            )
        except Exception as exc:
            raise JWTAuthError("invalid websocket JWT") from exc

        if not isinstance(claims, dict):
            raise JWTAuthError("invalid websocket JWT claims")
        return claims

    def _get_context(self) -> JWTVerificationContext:
        if self._context is None:
            self._context = self._load_context()
        return self._context

    def _load_context(self) -> JWTVerificationContext:
        values = app_settings.jwt_auth_env()
        public_key = values.public_key
        public_key_file = values.public_key_file
        if public_key is None and public_key_file is not None:
            key_path = public_key_file
            public_key = key_path.read_text(encoding="utf-8")
        elif public_key is None:
            raise JWTAuthError(
                "missing Ed25519 public key; set "
                "HLGAM_WS_JWT_PUBLIC_KEY or HLGAM_WS_JWT_PUBLIC_KEY_FILE "
                "in environment"
            )

        return JWTVerificationContext(
            public_key=public_key,
            audience=values.audience,
            issuer=values.issuer,
            algorithms=self.settings.algorithms,
            required_claims=self.settings.required_claims,
            leeway_seconds=self.settings.leeway_seconds,
        )
