from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from hlgam.broker.websocket_broker import (
    WebSocketGestureBroker,
    extract_bearer_token,
    extract_token_message,
)
from hlgam.gestures.engine import GestureEngineConfig
from hlgam.gestures.pinch_drag import PinchDragConfig
from hlgam.ml.inference.motion import MotionModelConfig
from hlgam.ml.inference.static_pose import StaticPoseModelConfig
from hlgam.models import Frame
from hlgam.processing.feature_extractor import FeatureExtractor
from hlgam.processing.normalization import HandNormalizer
from hlgam.recognition import LearnedModelsConfig, RecognitionConfig
from hlgam.security.jwt_auth import JWTAuthError, JWTAuthSettings, JWTVerifier
from tests.helpers import synthetic_hand


class FakeVerifier:
    def decode(self, token: str) -> dict[str, Any]:
        if token != "valid-token":
            raise JWTAuthError("invalid token")
        return {"sub": "client-1"}


class FakeWebSocket:
    def __init__(self, headers: dict[str, str] | None = None, messages: list[str] | None = None) -> None:
        self.request_headers = headers or {}
        self.messages = list(messages or [])
        self.sent: list[str] = []
        self.close_code: int | None = None
        self.close_reason: str | None = None
        self.closed = asyncio.Event()

    async def recv(self) -> str:
        if not self.messages:
            await asyncio.sleep(10)
        return self.messages.pop(0)

    async def send(self, payload: str) -> None:
        self.sent.append(payload)

    async def close(self, code: int, reason: str) -> None:
        self.close_code = code
        self.close_reason = reason
        self.closed.set()

    async def wait_closed(self) -> None:
        await self.closed.wait()


def pinch_features(timestamp: float = 0.0):
    raw = synthetic_hand(timestamp=timestamp, normalized_pinch_distance=0.08)
    normalizer = HandNormalizer()
    extractor = FeatureExtractor()
    return extractor.extract(normalizer.normalize(raw), raw_landmarks=raw)


def add_fake_client(broker: WebSocketGestureBroker, websocket: FakeWebSocket):
    state = broker._create_client_state(websocket, {})
    broker._clients.add(websocket)
    broker._client_states[websocket] = state
    return state


def hand_message(timestamp: float = 0.0) -> dict[str, Any]:
    hand = synthetic_hand(timestamp=timestamp, normalized_pinch_distance=0.08)
    return {
        "landmarks": [point.to_dict() for point in hand.landmarks],
        "timestamp": hand.timestamp,
        "handedness": hand.handedness,
        "score": hand.score,
        "image_size": hand.image_size,
    }


async def wait_for_client(broker: WebSocketGestureBroker, websocket: FakeWebSocket) -> None:
    for _ in range(25):
        if websocket in broker._client_states:
            return
        await asyncio.sleep(0)
    raise AssertionError("websocket was not registered as a client")


async def wait_for_sent(websocket: FakeWebSocket, expected: str) -> None:
    for _ in range(25):
        if any(expected in payload for payload in websocket.sent):
            return
        await asyncio.sleep(0)
    raise AssertionError(f"expected websocket payload containing {expected!r}")


def test_extract_bearer_token() -> None:
    assert extract_bearer_token("Bearer abc.def") == "abc.def"
    assert extract_bearer_token("bearer token") == "token"
    assert extract_bearer_token("Basic token") is None
    assert extract_bearer_token(None) is None


def test_extract_token_message() -> None:
    assert extract_token_message('{"token": "abc"}') == "abc"
    assert extract_token_message(b'{"token": "abc"}') == "abc"
    assert extract_token_message('{"token": ""}') is None
    assert extract_token_message('{"not_token": "abc"}') is None
    assert extract_token_message("not-json") is None


def test_jwt_verifier_decodes_eddsa_token_from_env_public_key(tmp_path: Path) -> None:
    jwt = pytest.importorskip("jwt")
    cryptography = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")

    private_key = cryptography.Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    token = jwt.encode(
        {"sub": "client-1", "aud": "gesture-clients", "iss": "hlgam-tests", "exp": int(time.time()) + 60},
        private_pem,
        algorithm="EdDSA",
    )
    escaped_public_key = public_pem.decode("utf-8").replace("\n", "\\n")
    env_file = tmp_path / ".env"
    env_file.write_text(
        f'HLGAM_WS_JWT_PUBLIC_KEY="{escaped_public_key}"\n'
        "HLGAM_WS_JWT_AUDIENCE=gesture-clients\n"
        "HLGAM_WS_JWT_ISSUER=hlgam-tests\n",
        encoding="utf-8",
    )

    claims = JWTVerifier(JWTAuthSettings()).decode(token)

    assert claims["sub"] == "client-1"


def test_websocket_handler_processes_features_only_after_bearer_auth() -> None:
    async def scenario() -> None:
        default_config = GestureEngineConfig(
            pinch_drag=PinchDragConfig(enabled=True, distance_threshold=0.10, release_threshold=0.16, hold_ms=0),
            gesture_mapping={"PINCH_START": "AUTH_CLICK"},
        )
        broker = WebSocketGestureBroker(verifier=FakeVerifier(), default_engine_config=default_config)
        websocket = FakeWebSocket(headers={"Authorization": "Bearer valid-token"})
        handler_task = asyncio.create_task(broker._handler(websocket))

        for _ in range(10):
            if websocket in broker._clients:
                break
            await asyncio.sleep(0)

        assert websocket in broker._clients
        await broker.process_features(pinch_features())
        websocket.closed.set()
        await handler_task

        assert len(websocket.sent) == 1
        assert "PINCH_START" in websocket.sent[0]
        assert "AUTH_CLICK" in websocket.sent[0]

    asyncio.run(scenario())


def test_websocket_handler_accepts_first_message_token() -> None:
    async def scenario() -> None:
        broker = WebSocketGestureBroker(verifier=FakeVerifier())
        websocket = FakeWebSocket(messages=['{"token": "valid-token"}'])

        auth_result = await broker._authenticate(websocket)

        assert auth_result is not None
        assert auth_result.claims == {"sub": "client-1"}
        assert websocket.close_code is None

    asyncio.run(scenario())


def test_websocket_handler_rejects_invalid_token_message() -> None:
    async def scenario() -> None:
        broker = WebSocketGestureBroker(verifier=FakeVerifier())
        websocket = FakeWebSocket(messages=['{"token": "bad-token"}'])

        auth_result = await broker._authenticate(websocket)

        assert auth_result is None
        assert websocket.close_code == 1008
        assert websocket not in broker._clients

    asyncio.run(scenario())


def test_websocket_client_without_config_uses_default_engine_config() -> None:
    async def scenario() -> None:
        default_config = GestureEngineConfig(
            pinch_drag=PinchDragConfig(enabled=True, distance_threshold=0.10, release_threshold=0.16, hold_ms=0),
            gesture_mapping={"PINCH_START": "DEFAULT_CLICK"},
        )
        broker = WebSocketGestureBroker(require_jwt=False, default_engine_config=default_config)
        websocket = FakeWebSocket()
        handler_task = asyncio.create_task(broker._handler(websocket))
        await wait_for_client(broker, websocket)

        await broker.process_features(pinch_features())
        websocket.closed.set()
        await handler_task

        assert any("PINCH_START" in payload and "DEFAULT_CLICK" in payload for payload in websocket.sent)

    asyncio.run(scenario())


def test_websocket_initial_message_can_provide_client_gesture_config() -> None:
    async def scenario() -> None:
        default_config = GestureEngineConfig(pinch_drag=PinchDragConfig(enabled=False))
        broker = WebSocketGestureBroker(verifier=FakeVerifier(), default_engine_config=default_config)
        websocket = FakeWebSocket(
            messages=[
                '{'
                '"token":"valid-token",'
                '"gestures":{"pinch_drag":{"enabled":true,"distance_threshold":0.10,"hold_ms":0}},'
                '"gesture_mapping":{"PINCH_START":"CLIENT_CLICK"}'
                '}'
            ]
        )
        handler_task = asyncio.create_task(broker._handler(websocket))
        await wait_for_client(broker, websocket)

        await broker.process_features(pinch_features())
        websocket.closed.set()
        await handler_task

        assert any("config_updated" in payload for payload in websocket.sent)
        assert any("PINCH_START" in payload and "CLIENT_CLICK" in payload for payload in websocket.sent)

    asyncio.run(scenario())


def test_websocket_client_can_update_gesture_config_during_stream() -> None:
    async def scenario() -> None:
        default_config = GestureEngineConfig(pinch_drag=PinchDragConfig(enabled=False))
        broker = WebSocketGestureBroker(require_jwt=False, default_engine_config=default_config)
        websocket = FakeWebSocket(
            messages=[
                '{'
                '"type":"update_config",'
                '"gestures":{"pinch_drag":{"enabled":true,"distance_threshold":0.10,"hold_ms":0}},'
                '"gesture_mapping":{"PINCH_START":"LIVE_CLICK"}'
                '}'
            ]
        )
        handler_task = asyncio.create_task(broker._handler(websocket))
        await wait_for_client(broker, websocket)
        await wait_for_sent(websocket, "config_updated")

        await broker.process_features(pinch_features())
        websocket.closed.set()
        await handler_task

        assert any("PINCH_START" in payload and "LIVE_CLICK" in payload for payload in websocket.sent)

    asyncio.run(scenario())


def test_websocket_client_can_update_recognition_pipeline_config() -> None:
    async def scenario() -> None:
        default_config = GestureEngineConfig()
        broker = WebSocketGestureBroker(require_jwt=False, default_engine_config=default_config)
        websocket = FakeWebSocket(
            messages=[
                '{'
                '"type":"update_config",'
                '"recognition":{'
                '"fsm":{"enabled":false},'
                '"models":{"static_pose":{'
                '"enabled":true,'
                '"pose":{"enabled":false},'
                '"finger_count":{"enabled":true},'
                '"finger_pointing":{"enabled":false},'
                '"flat_hand":{"enabled":false}'
                '},'
                '"motion":{"enabled":true}'
                '}'
                '}'
                '}'
            ]
        )
        handler_task = asyncio.create_task(broker._handler(websocket))
        await wait_for_client(broker, websocket)
        await wait_for_sent(websocket, "config_updated")
        state = broker._client_states[websocket]

        websocket.closed.set()
        await handler_task

        assert state.config.recognition.fsm.enabled is False
        assert state.config.recognition.models.static_pose.enabled is True
        assert state.config.recognition.models.static_pose.pose.enabled is False
        assert state.config.recognition.models.static_pose.finger_count.enabled is True
        assert state.config.recognition.models.static_pose.finger_pointing.enabled is False
        assert state.config.recognition.models.static_pose.flat_hand.enabled is False
        assert state.config.recognition.models.motion.enabled is True
        assert any('"recognition"' in payload for payload in websocket.sent)

    asyncio.run(scenario())


def test_websocket_client_can_use_top_level_models_shortcut() -> None:
    async def scenario() -> None:
        default_config = GestureEngineConfig(
            recognition=RecognitionConfig(
                models=LearnedModelsConfig(
                    static_pose=StaticPoseModelConfig(enabled=False),
                    motion=MotionModelConfig(enabled=False),
                )
            )
        )
        broker = WebSocketGestureBroker(require_jwt=False, default_engine_config=default_config)
        websocket = FakeWebSocket(
            messages=[
                '{'
                '"type":"update_config",'
                '"models":{'
                '"static_pose":{"enabled":true,"pose":{"enabled":false}},'
                '"motion":{"enabled":true}'
                '}'
                '}'
            ]
        )
        handler_task = asyncio.create_task(broker._handler(websocket))
        await wait_for_client(broker, websocket)
        await wait_for_sent(websocket, "config_updated")
        state = broker._client_states[websocket]

        websocket.closed.set()
        await handler_task

        assert state.config.recognition.models.static_pose.enabled is True
        assert state.config.recognition.models.static_pose.pose.enabled is False
        assert state.config.recognition.models.static_pose.finger_count.enabled is True
        assert state.config.recognition.models.motion.enabled is True

    asyncio.run(scenario())


def test_websocket_invalid_live_config_does_not_replace_existing_config() -> None:
    async def scenario() -> None:
        default_config = GestureEngineConfig(
            pinch_drag=PinchDragConfig(enabled=True, distance_threshold=0.10, release_threshold=0.16, hold_ms=0),
            gesture_mapping={"PINCH_START": "DEFAULT_CLICK"},
        )
        broker = WebSocketGestureBroker(require_jwt=False, default_engine_config=default_config)
        websocket = FakeWebSocket(
            messages=['{"type":"update_config","gestures":{"pinch_drag":{"hold_ms":-1}}}']
        )
        handler_task = asyncio.create_task(broker._handler(websocket))
        await wait_for_client(broker, websocket)
        await wait_for_sent(websocket, "invalid_config")

        await broker.process_features(pinch_features())
        websocket.closed.set()
        await handler_task

        assert any("PINCH_START" in payload and "DEFAULT_CLICK" in payload for payload in websocket.sent)

    asyncio.run(scenario())


def test_websocket_client_can_disable_individual_gesture_event() -> None:
    async def scenario() -> None:
        default_config = GestureEngineConfig(
            pinch_drag=PinchDragConfig(enabled=True, distance_threshold=0.10, release_threshold=0.16, hold_ms=0)
        )
        broker = WebSocketGestureBroker(require_jwt=False, default_engine_config=default_config)
        websocket = FakeWebSocket(messages=['{"type":"update_config","gesture_events":{"PINCH_START":false}}'])
        handler_task = asyncio.create_task(broker._handler(websocket))
        await wait_for_client(broker, websocket)
        await wait_for_sent(websocket, "config_updated")

        await broker.process_features(pinch_features())
        websocket.closed.set()
        await handler_task

        assert not any('"gesture":"PINCH_START"' in payload for payload in websocket.sent)
        assert any('"PINCH_START":false' in payload for payload in websocket.sent)

    asyncio.run(scenario())


def test_websocket_clients_receive_events_only_from_selected_video_source() -> None:
    async def scenario() -> None:
        default_config = GestureEngineConfig(
            pinch_drag=PinchDragConfig(enabled=True, distance_threshold=0.10, release_threshold=0.16, hold_ms=0)
        )
        broker = WebSocketGestureBroker(require_jwt=False, default_engine_config=default_config)
        internal_websocket = FakeWebSocket()
        remote_websocket = FakeWebSocket()
        producer_websocket = FakeWebSocket()
        internal_state = add_fake_client(broker, internal_websocket)
        remote_state = add_fake_client(broker, remote_websocket)
        producer_state = add_fake_client(broker, producer_websocket)

        await broker._start_video_input(producer_state, {"type": "start_video_input", "source_id": "phone"})
        await broker._select_video_source(remote_state, {"type": "select_video_source", "source_id": "phone"})

        await broker.process_source_hands(
            "internal_camera",
            [synthetic_hand(timestamp=0.0, normalized_pinch_distance=0.08)],
        )
        await broker.process_source_hands("phone", [synthetic_hand(timestamp=0.0, normalized_pinch_distance=0.08)])

        internal_events = [json.loads(payload) for payload in internal_websocket.sent if "PINCH_START" in payload]
        remote_events = [json.loads(payload) for payload in remote_websocket.sent if "PINCH_START" in payload]
        assert internal_state.selected_source_id == "internal_camera"
        assert remote_state.selected_source_id == "phone"
        assert [event["source_id"] for event in internal_events] == ["internal_camera"]
        assert [event["source_id"] for event in remote_events] == ["phone"]

    asyncio.run(scenario())


def test_websocket_input_feed_can_stream_landmarks_for_detection() -> None:
    async def scenario() -> None:
        default_config = GestureEngineConfig(
            pinch_drag=PinchDragConfig(enabled=True, distance_threshold=0.10, release_threshold=0.16, hold_ms=0)
        )
        broker = WebSocketGestureBroker(require_jwt=False, default_engine_config=default_config)
        producer_websocket = FakeWebSocket()
        consumer_websocket = FakeWebSocket()
        producer_state = add_fake_client(broker, producer_websocket)
        consumer_state = add_fake_client(broker, consumer_websocket)

        await broker._start_video_input(producer_state, {"type": "start_video_input", "source_id": "phone"})
        await broker._select_video_source(consumer_state, {"type": "select_video_source", "source_id": "phone"})
        await broker._ingest_video_frame(
            producer_state,
            {"type": "video_frame", "source_id": "phone", "hands": [hand_message(timestamp=1.0)]},
        )

        assert any("video_input_started" in payload for payload in producer_websocket.sent)
        consumer_events = [json.loads(payload) for payload in consumer_websocket.sent if "PINCH_START" in payload]
        assert len(consumer_events) == 1
        assert consumer_events[0]["source_id"] == "phone"

    asyncio.run(scenario())


def test_websocket_output_feed_streams_selected_source_landmarks_and_features() -> None:
    async def scenario() -> None:
        broker = WebSocketGestureBroker(require_jwt=False)
        producer_websocket = FakeWebSocket()
        viewer_websocket = FakeWebSocket()
        producer_state = add_fake_client(broker, producer_websocket)
        viewer_state = add_fake_client(broker, viewer_websocket)

        await broker._start_video_input(producer_state, {"type": "start_video_input", "source_id": "phone"})
        await broker._start_video_output(
            viewer_state,
            {
                "type": "start_video_output",
                "source_id": "phone",
                "include_frame": False,
                "include_landmarks": True,
                "include_features": True,
            },
        )
        await broker._ingest_video_frame(
            producer_state,
            {"type": "video_frame", "source_id": "phone", "hands": [hand_message(timestamp=1.0)]},
        )

        frames = [json.loads(payload) for payload in viewer_websocket.sent if '"type":"video_frame"' in payload]
        assert viewer_state.mode == "output"
        assert len(frames) == 1
        assert frames[0]["source_id"] == "phone"
        assert frames[0]["frame"] is None
        assert len(frames[0]["hands"]) == 1
        assert len(frames[0]["features"]) == 1

    asyncio.run(scenario())


def test_websocket_output_feed_streams_encoded_frames() -> None:
    async def scenario() -> None:
        numpy = pytest.importorskip("numpy")
        broker = WebSocketGestureBroker(require_jwt=False)
        viewer_websocket = FakeWebSocket()
        viewer_state = add_fake_client(broker, viewer_websocket)

        await broker._start_video_output(
            viewer_state,
            {"type": "start_video_output", "source_id": "internal_camera", "include_frame": True},
        )
        frame = Frame(image=numpy.zeros((2, 3, 3), dtype=numpy.uint8), timestamp=1.25, index=7)
        await broker.process_source_frame("internal_camera", frame, [])

        frames = [json.loads(payload) for payload in viewer_websocket.sent if '"type":"video_frame"' in payload]
        assert len(frames) == 1
        assert frames[0]["frame"]["encoding"] == "jpeg_base64"
        assert frames[0]["frame"]["width"] == 3
        assert frames[0]["frame"]["height"] == 2
        assert frames[0]["frame"]["index"] == 7
        assert isinstance(frames[0]["frame"]["data"], str)

    asyncio.run(scenario())
