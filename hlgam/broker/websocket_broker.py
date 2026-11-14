from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hlgam.broker.client_config import ClientConfigError, config_payload, is_config_message, merge_client_engine_config
from hlgam.broker.video_feed import (
    INTERNAL_VIDEO_SOURCE_ID,
    VideoFeedError,
    VideoOutputOptions,
    VideoSourceState,
    decode_frame_from_message,
    encode_frame_payload,
    features_payload,
    hand_payload,
    metadata_value,
    parse_hands_from_message,
    string_value,
    validate_source_id,
)
from hlgam.gestures.engine import GestureEngine, GestureEngineConfig
from hlgam.models import Frame, GestureEvent, GestureFeatures, HandLandmarks
from hlgam.security.jwt_auth import JWTAuthError, JWTAuthSettings, JWTVerifier


POLICY_VIOLATION = 1008


@dataclass
class AuthResult:
    claims: dict[str, Any]
    initial_payload: dict[str, Any] | None = None


@dataclass
class ClientState:
    websocket: Any
    claims: dict[str, Any]
    config: GestureEngineConfig
    engine: GestureEngine
    mode: str = "events"
    selected_source_id: str = INTERNAL_VIDEO_SOURCE_ID
    input_source_id: str | None = None
    video_output: VideoOutputOptions = field(default_factory=VideoOutputOptions)


class WebSocketGestureBroker:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        *,
        require_jwt: bool = True,
        jwt_env_file: str | Path = ".env",
        jwt_auth_timeout_seconds: float = 5.0,
        default_engine_config: GestureEngineConfig | None = None,
        verifier: JWTVerifier | None = None,
        tracker_settings: Any | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.require_jwt = require_jwt
        self.jwt_auth_timeout_seconds = jwt_auth_timeout_seconds
        self.handles_features = True
        self._default_engine_config = default_engine_config or GestureEngineConfig()
        self._verifier = verifier or JWTVerifier(JWTAuthSettings())
        self._tracker_settings = tracker_settings
        self._clients: set = set()
        self._client_claims: dict[object, dict[str, Any]] = {}
        self._client_states: dict[object, ClientState] = {}
        self._sources: dict[str, VideoSourceState] = {
            INTERNAL_VIDEO_SOURCE_ID: VideoSourceState(
                source_id=INTERNAL_VIDEO_SOURCE_ID,
                label="Internal camera",
                kind="internal",
                tracker_settings=tracker_settings,
            )
        }
        self._server = None

    async def start(self) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("WebSocket broadcasting requires the websockets optional dependency") from exc
        self._server = await websockets.serve(self._handler, self.host, self.port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        for source in self._sources.values():
            source.close()

    async def publish(self, event: GestureEvent) -> None:
        if self.handles_features:
            return
        if not self._clients:
            return
        payload = event.to_json()
        failed = []
        for client in tuple(self._clients):
            try:
                await client.send(payload)
            except Exception:
                failed.append(client)
        for client in failed:
            self._clients.discard(client)

    async def process_features(self, features: GestureFeatures) -> None:
        await self.process_feature_batch([features])

    async def process_feature_batch(
        self,
        features_list: list[GestureFeatures],
        source_id: str = INTERNAL_VIDEO_SOURCE_ID,
        frame: Frame | None = None,
        hands: list[HandLandmarks] | None = None,
    ) -> None:
        source = self._ensure_source(source_id)
        source.update_frame_metadata(frame)
        if frame is not None or hands is not None:
            await self._stream_video_frame(source, frame, hands or [], features_list)
        await self._publish_source_features(source, features_list)

    async def process_source_frame(
        self,
        source_id: str,
        frame: Frame | None,
        hands: list[HandLandmarks],
    ) -> None:
        source = self._ensure_source(source_id)
        source.update_frame_metadata(frame)
        features_list = source.processor.extract_features(hands)
        await self._stream_video_frame(source, frame, hands, features_list)
        await self._publish_source_features(source, features_list)

    async def process_source_hands(
        self,
        source_id: str,
        hands: list[HandLandmarks],
        frame: Frame | None = None,
    ) -> None:
        await self.process_source_frame(source_id, frame, hands)

    async def _publish_source_features(self, source: VideoSourceState, features_list: list[GestureFeatures]) -> None:
        failed = []
        for state in tuple(self._client_states.values()):
            if state.mode != "events" or state.selected_source_id != source.source_id:
                continue
            if not features_list:
                state.engine.reset()
                continue
            try:
                events = state.engine.process_many(features_list)
            except Exception as exc:
                try:
                    await send_control_message(
                        state.websocket,
                        {"type": "error", "error": "recognition_runtime", "message": str(exc)},
                    )
                except Exception:
                    failed.append(state.websocket)
                continue
            for event in events:
                try:
                    await state.websocket.send(source_event_json(event, source))
                except Exception:
                    failed.append(state.websocket)
                    break
        for websocket in failed:
            self._remove_client(websocket)

    async def _handler(self, websocket) -> None:
        auth_result = await self._authenticate(websocket)
        if auth_result is None:
            return

        state = self._create_client_state(websocket, auth_result.claims)
        initial_command_payload = None
        if auth_result.initial_payload is not None:
            if is_config_message(auth_result.initial_payload):
                if not await self._apply_client_config(state, auth_result.initial_payload):
                    await close_policy_violation(websocket, "invalid initial websocket client config")
                    return
            elif command_name(auth_result.initial_payload):
                initial_command_payload = auth_result.initial_payload

        self._clients.add(websocket)
        self._client_claims[websocket] = auth_result.claims
        self._client_states[websocket] = state
        if initial_command_payload is not None:
            await self._handle_client_command(state, initial_command_payload)

        listener_task = asyncio.create_task(self._listen_for_client_commands(state))
        try:
            await websocket.wait_closed()
        finally:
            listener_task.cancel()
            try:
                await listener_task
            except asyncio.CancelledError:
                pass
            self._remove_client(websocket)

    async def _authenticate(self, websocket) -> AuthResult | None:
        if not self.require_jwt:
            return AuthResult(claims={})

        token = extract_bearer_token(get_authorization_header(websocket))
        initial_payload = None
        if token is None:
            initial_payload = await self._read_first_message_payload(websocket)
            if initial_payload is None:
                return None
            token = extract_token_message(initial_payload)
        if token is None:
            await close_policy_violation(websocket, "first websocket message must include a token")
            return None

        try:
            return AuthResult(claims=self._verifier.decode(token), initial_payload=initial_payload)
        except JWTAuthError:
            await close_policy_violation(websocket, "invalid or expired websocket JWT")
            return None

    async def _read_first_message_payload(self, websocket) -> dict[str, Any] | None:
        try:
            message = await asyncio.wait_for(websocket.recv(), timeout=self.jwt_auth_timeout_seconds)
        except TimeoutError:
            await close_policy_violation(websocket, "websocket JWT auth token was not provided in time")
            return None
        except Exception:
            await close_policy_violation(websocket, "websocket JWT auth token could not be read")
            return None

        payload = parse_json_object_message(message)
        if payload is None:
            await close_policy_violation(websocket, "first websocket message must be a JSON token payload")
        return payload

    def _create_client_state(self, websocket, claims: dict[str, Any]) -> ClientState:
        return ClientState(
            websocket=websocket,
            claims=claims,
            config=self._default_engine_config,
            engine=GestureEngine(self._default_engine_config),
        )

    async def _listen_for_client_commands(self, state: ClientState) -> None:
        while True:
            try:
                message = await state.websocket.recv()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._remove_client(state.websocket)
                return

            payload = parse_json_object_message(message)
            if payload is None:
                await send_control_message(
                    state.websocket,
                    {"type": "error", "error": "invalid_json", "message": "client command must be a JSON object"},
                )
                continue
            await self._handle_client_command(state, payload)

    async def _handle_client_command(self, state: ClientState, payload: dict[str, Any]) -> None:
        command = command_name(payload)
        if is_config_message(payload):
            await self._apply_client_config(state, payload)
            return
        if command == "list_video_sources":
            await self._send_source_list(state.websocket)
            return
        if command in {"select_video_source", "switch_video_source"}:
            await self._select_video_source(state, payload)
            return
        if command in {"start_video_input", "register_video_input"}:
            await self._start_video_input(state, payload)
            return
        if command in {"stop_video_input", "unregister_video_input"}:
            await self._stop_video_input(state)
            return
        if command == "video_frame":
            await self._ingest_video_frame(state, payload)
            return
        if command in {"start_video_output", "watch_video_source"}:
            await self._start_video_output(state, payload)
            return
        if command in {"stop_video_output", "stop_watching_video"}:
            await self._stop_video_output(state)
            return
        await send_control_message(
            state.websocket,
            {"type": "error", "error": "unknown_command", "message": "unsupported websocket command"},
        )

    async def _select_video_source(self, state: ClientState, payload: dict[str, Any]) -> bool:
        try:
            source_id = validate_source_id(payload.get("source_id", payload.get("source")))
        except VideoFeedError as exc:
            await self._send_video_error(state.websocket, str(exc))
            return False
        if source_id not in self._sources:
            await self._send_video_error(state.websocket, f"unknown video source: {source_id}")
            return False
        state.selected_source_id = source_id
        state.engine.reset()
        await send_control_message(
            state.websocket,
            {"type": "source_selected", "source_id": source_id, "source": self._sources[source_id].info_payload()},
        )
        return True

    async def _start_video_input(self, state: ClientState, payload: dict[str, Any]) -> None:
        try:
            source_id = validate_source_id(payload.get("source_id") or default_source_id_for(state.websocket))
            if source_id == INTERNAL_VIDEO_SOURCE_ID:
                raise VideoFeedError(f"{INTERNAL_VIDEO_SOURCE_ID} is reserved for the internal camera")
            existing = self._sources.get(source_id)
            if existing is not None and existing.owner is not state.websocket:
                raise VideoFeedError(f"video source already exists: {source_id}")
            label = string_value(payload.get("label"), source_id, "label")
            metadata = metadata_value(payload.get("metadata"))
        except VideoFeedError as exc:
            await self._send_video_error(state.websocket, str(exc))
            return

        if state.input_source_id and state.input_source_id != source_id:
            self._remove_source(state.input_source_id)
        source = VideoSourceState(
            source_id=source_id,
            label=label,
            kind="remote",
            owner=state.websocket,
            metadata=metadata,
            tracker_settings=self._tracker_settings,
        )
        self._sources[source_id] = source
        state.mode = "input"
        state.input_source_id = source_id
        await send_control_message(
            state.websocket,
            {"type": "video_input_started", "source_id": source_id, "source": source.info_payload()},
        )
        await self._broadcast_source_list("video_sources_updated")

    async def _stop_video_input(self, state: ClientState) -> None:
        source_id = state.input_source_id
        if source_id is not None:
            self._remove_source(source_id)
            state.input_source_id = None
        state.mode = "events"
        await send_control_message(state.websocket, {"type": "video_input_stopped", "source_id": source_id})
        await self._broadcast_source_list("video_sources_updated")

    async def _ingest_video_frame(self, state: ClientState, payload: dict[str, Any]) -> None:
        source_id = payload.get("source_id") or state.input_source_id
        try:
            source_id = validate_source_id(source_id)
        except VideoFeedError as exc:
            await self._send_video_error(state.websocket, str(exc))
            return
        source = self._sources.get(source_id)
        if source is None or source.owner is not state.websocket:
            await self._send_video_error(state.websocket, f"websocket does not own video source: {source_id}")
            return

        try:
            frame = decode_frame_from_message(payload)
            hands = parse_hands_from_message(payload, frame)
            if hands is None:
                if frame is None:
                    raise VideoFeedError("video_frame must include frame.data or hands")
                hands = source.track(frame)
        except VideoFeedError as exc:
            await self._send_video_error(state.websocket, str(exc))
            return
        await self.process_source_frame(source_id, frame, hands)

    async def _start_video_output(self, state: ClientState, payload: dict[str, Any]) -> None:
        try:
            state.video_output = VideoOutputOptions.from_payload(payload)
        except VideoFeedError as exc:
            await self._send_video_error(state.websocket, str(exc))
            return
        if payload.get("source_id") is not None and not await self._select_video_source(state, payload):
            return
        state.mode = "output"
        await send_control_message(
            state.websocket,
            {
                "type": "video_output_started",
                "source_id": state.selected_source_id,
                "source": self._sources[state.selected_source_id].info_payload(),
                "include_frame": state.video_output.include_frame,
                "include_landmarks": state.video_output.include_landmarks,
                "include_features": state.video_output.include_features,
            },
        )

    async def _stop_video_output(self, state: ClientState) -> None:
        state.mode = "events"
        state.engine.reset()
        await send_control_message(
            state.websocket,
            {"type": "video_output_stopped", "source_id": state.selected_source_id},
        )

    async def _stream_video_frame(
        self,
        source: VideoSourceState,
        frame: Frame | None,
        hands: list[HandLandmarks],
        features_list: list[GestureFeatures],
    ) -> None:
        failed = []
        encoded_frames: dict[int, dict[str, Any]] = {}
        for state in tuple(self._client_states.values()):
            if state.mode != "output" or state.selected_source_id != source.source_id:
                continue
            try:
                frame_payload = None
                if frame is not None and state.video_output.include_frame:
                    quality = state.video_output.jpeg_quality
                    if quality not in encoded_frames:
                        encoded_frames[quality] = encode_frame_payload(frame, quality)
                    frame_payload = encoded_frames[quality]
                await state.websocket.send(json.dumps(
                    {
                        "type": "video_frame",
                        "source_id": source.source_id,
                        "source": source.info_payload(),
                        "timestamp": time.time(),
                        "frame": frame_payload,
                        "hands": [hand_payload(hand) for hand in hands]
                        if state.video_output.include_landmarks else None,
                        "features": [features_payload(features) for features in features_list]
                        if state.video_output.include_features else None,
                    },
                    separators=(",", ":"),
                ))
            except Exception:
                failed.append(state.websocket)
        for websocket in failed:
            self._remove_client(websocket)

    async def _send_source_list(self, websocket) -> None:
        await send_control_message(
            websocket,
            {"type": "video_sources", "sources": [source.info_payload() for source in self._sources.values()]},
        )

    async def _broadcast_source_list(self, message_type: str) -> None:
        payload = {"type": message_type, "sources": [source.info_payload() for source in self._sources.values()]}
        for state in tuple(self._client_states.values()):
            await send_control_message(state.websocket, payload)

    async def _send_video_error(self, websocket, message: str) -> None:
        await send_control_message(websocket, {"type": "error", "error": "video_feed", "message": message})

    async def _apply_client_config(self, state: ClientState, payload: dict[str, Any]) -> bool:
        if not is_config_message(payload):
            return True
        try:
            next_config = merge_client_engine_config(state.config, payload)
        except ClientConfigError as exc:
            await send_control_message(
                state.websocket,
                {"type": "error", "error": "invalid_config", "message": str(exc)},
            )
            return False

        state.config = next_config
        state.engine = GestureEngine(next_config)
        await send_control_message(
            state.websocket,
            {
                "type": "config_updated",
                "timestamp": time.time(),
                **config_payload(next_config),
            },
        )
        return True

    def _remove_client(self, websocket) -> None:
        state = self._client_states.get(websocket)
        removed_source = state.input_source_id if state is not None else None
        if state is not None and state.input_source_id is not None:
            self._remove_source(state.input_source_id)
        self._clients.discard(websocket)
        self._client_claims.pop(websocket, None)
        self._client_states.pop(websocket, None)
        if removed_source is not None:
            self._schedule_source_list_broadcast()

    def _ensure_source(self, source_id: str) -> VideoSourceState:
        source_id = validate_source_id(source_id)
        source = self._sources.get(source_id)
        if source is not None:
            return source
        source = VideoSourceState(
            source_id=source_id,
            label=source_id,
            kind="external",
            tracker_settings=self._tracker_settings,
        )
        self._sources[source_id] = source
        return source

    def _remove_source(self, source_id: str) -> None:
        if source_id == INTERNAL_VIDEO_SOURCE_ID:
            return
        source = self._sources.pop(source_id, None)
        if source is not None:
            source.close()
        for state in self._client_states.values():
            if state.selected_source_id == source_id:
                state.selected_source_id = INTERNAL_VIDEO_SOURCE_ID
                state.engine.reset()

    def _schedule_source_list_broadcast(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._broadcast_source_list("video_sources_updated"))

    def publish_background(self, event: GestureEvent) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.publish(event))
            return
        loop.create_task(self.publish(event))


def get_authorization_header(websocket) -> str | None:
    headers = getattr(websocket, "request_headers", None)
    if headers is None:
        request = getattr(websocket, "request", None)
        headers = getattr(request, "headers", None)
    if headers is None:
        return None

    for header_name in ("Authorization", "authorization"):
        try:
            value = headers.get(header_name)
        except AttributeError:
            value = None
        if value:
            return str(value)

    try:
        items = headers.items()
    except AttributeError:
        items = headers
    for name, value in items:
        if str(name).casefold() == "authorization":
            return str(value)
    return None


def extract_bearer_token(header_value: str | None) -> str | None:
    if not header_value:
        return None
    parts = header_value.strip().split(None, 1)
    if len(parts) != 2 or parts[0].casefold() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def extract_token_message(message: str | bytes | dict[str, Any]) -> str | None:
    payload = parse_json_object_message(message)
    if payload is None:
        return None
    token = payload.get("token")
    if not isinstance(token, str):
        return None
    token = token.strip()
    return token or None


def parse_json_object_message(message: str | bytes | dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(message, dict):
        return message
    if isinstance(message, bytes):
        try:
            message = message.decode("utf-8")
        except UnicodeDecodeError:
            return None
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def command_name(payload: dict[str, Any]) -> str:
    command = payload.get("type", payload.get("command"))
    return command.casefold() if isinstance(command, str) else ""


def default_source_id_for(websocket: Any) -> str:
    return f"client:{id(websocket):x}"


def source_event_json(event: GestureEvent, source: VideoSourceState) -> str:
    payload = event.to_dict()
    metadata = dict(payload.get("metadata") or {})
    metadata.setdefault("source_id", source.source_id)
    payload["metadata"] = metadata
    payload["source_id"] = source.source_id
    payload["source"] = source.info_payload()
    return json.dumps(payload, separators=(",", ":"))


async def close_policy_violation(websocket, reason: str) -> None:
    try:
        await websocket.close(code=POLICY_VIOLATION, reason=reason)
    except Exception:
        return


async def send_control_message(websocket, payload: dict[str, Any]) -> None:
    try:
        await websocket.send(json.dumps(payload, separators=(",", ":")))
    except Exception:
        return
