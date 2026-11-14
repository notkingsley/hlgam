from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from hlgam.broker.osc_broker import OSCGestureBroker
from hlgam.broker.websocket_broker import WebSocketGestureBroker
from hlgam.capture.camera import CameraSettings, OpenCVCamera
from hlgam.config import HLGAMConfig, load_config
from hlgam.gestures.engine import GestureEngine, GestureEngineConfig
from hlgam.models import GestureEvent, HandLandmarks
from hlgam.models import GestureFeatures
from hlgam.processing.feature_extractor import FeatureExtractor
from hlgam.processing.normalization import HandNormalizer
from hlgam.processing.smoothing import ExponentialLandmarkSmoother
from hlgam.tracking.hand_tracker import MediaPipeHandTracker

GestureCallback = Callable[[GestureEvent], None]


class GestureFramework:
    def __init__(self, config: HLGAMConfig | None = None) -> None:
        self.config = config or HLGAMConfig()
        self.normalizer = HandNormalizer()
        self.smoother = ExponentialLandmarkSmoother()
        self.extractor = FeatureExtractor()
        self._multi_extractors: dict[str, FeatureExtractor] = {}
        self.engine = GestureEngine(self._engine_config())
        self._callbacks: dict[str, list[GestureCallback]] = defaultdict(list)
        self._brokers: list[object] = []

    @classmethod
    def from_config_file(cls, path: str | Path) -> GestureFramework:
        return cls(load_config(path))

    def on(self, gesture: str, callback: GestureCallback) -> None:
        self._callbacks[gesture].append(callback)

    def on_any(self, callback: GestureCallback) -> None:
        self.on("*", callback)

    def add_broker(self, broker: object) -> None:
        self._brokers.append(broker)

    def configure_brokers(self) -> None:
        if self.config.broker.websocket.enabled:
            websocket = self.config.broker.websocket
            self.add_broker(
                WebSocketGestureBroker(
                    websocket.host,
                    websocket.port,
                    require_jwt=websocket.require_jwt,
                    jwt_env_file=websocket.jwt_env_file,
                    jwt_auth_timeout_seconds=websocket.jwt_auth_timeout_seconds,
                    default_engine_config=GestureEngineConfig(
                        pinch_drag=self.config.gestures.pinch_drag,
                        swipe_motion=self.config.gestures.swipe_motion,
                        zoom=self.config.gestures.zoom,
                        pose=self.config.gestures.pose,
                        finger_count=self.config.gestures.finger_count,
                        finger_pointing=self.config.gestures.finger_pointing,
                        flat_hand=self.config.gestures.flat_hand,
                        flick=self.config.gestures.flick,
                        rotation=self.config.gestures.rotation,
                        scroll=self.config.gestures.scroll,
                        motion=self.config.gestures.motion,
                        air_draw=self.config.gestures.air_draw,
                        two_hand=self.config.gestures.two_hand,
                        recognition=self.config.recognition,
                        gesture_events=self.config.gesture_events,
                        gesture_mapping=self.config.gesture_mapping,
                    ),
                )
            )
        if self.config.broker.osc.enabled:
            osc = self.config.broker.osc
            self.add_broker(OSCGestureBroker(osc.host, osc.port))

    async def start_async_brokers(self) -> None:
        for broker in self._brokers:
            start = getattr(broker, "start", None)
            if start is None:
                continue
            result = start()
            if inspect.isawaitable(result):
                await result

    async def stop_async_brokers(self) -> None:
        for broker in self._brokers:
            stop = getattr(broker, "stop", None)
            if stop is None:
                continue
            result = stop()
            if inspect.isawaitable(result):
                await result

    def process_landmarks(self, hand: HandLandmarks) -> list[GestureEvent]:
        smoothed = self.smoother.smooth(hand)
        normalized = self.normalizer.normalize(smoothed)
        features = self.extractor.extract(normalized, raw_landmarks=smoothed)
        self._dispatch_features(features)
        events = self.engine.process(features)
        for event in events:
            self._dispatch(event)
        return events

    def process_hands(self, hands: Sequence[HandLandmarks]) -> list[GestureEvent]:
        if not hands:
            self.reset()
            return []

        features_list: list[GestureFeatures] = []
        for index, hand in enumerate(hands):
            smoothed = self.smoother.smooth(hand)
            normalized = self.normalizer.normalize(smoothed)
            extractor = self._extractor_for(smoothed, index)
            features_list.append(extractor.extract(normalized, raw_landmarks=smoothed))

        self._dispatch_feature_batch(features_list)
        events = self.engine.process_many(features_list)
        for event in events:
            self._dispatch(event)
        return events

    def _dispatch_features(self, features: GestureFeatures) -> None:
        for broker in self._brokers:
            process_features = getattr(broker, "process_features", None)
            if process_features is None:
                continue
            result = process_features(features)
            if inspect.isawaitable(result):
                self._schedule(result)

    def _dispatch_feature_batch(self, features_list: list[GestureFeatures]) -> None:
        for broker in self._brokers:
            process_feature_batch = getattr(broker, "process_feature_batch", None)
            if process_feature_batch is None:
                for features in features_list:
                    process_features = getattr(broker, "process_features", None)
                    if process_features is None:
                        continue
                    result = process_features(features)
                    if inspect.isawaitable(result):
                        self._schedule(result)
                continue
            result = process_feature_batch(features_list)
            if inspect.isawaitable(result):
                self._schedule(result)

    def reset(self) -> None:
        self.smoother.reset()
        self.extractor.reset()
        self._multi_extractors.clear()
        self.engine.reset()

    def _engine_config(self) -> GestureEngineConfig:
        return GestureEngineConfig(
            pinch_drag=self.config.gestures.pinch_drag,
            swipe_motion=self.config.gestures.swipe_motion,
            zoom=self.config.gestures.zoom,
            pose=self.config.gestures.pose,
            finger_count=self.config.gestures.finger_count,
            finger_pointing=self.config.gestures.finger_pointing,
            flat_hand=self.config.gestures.flat_hand,
            flick=self.config.gestures.flick,
            rotation=self.config.gestures.rotation,
            scroll=self.config.gestures.scroll,
            motion=self.config.gestures.motion,
            air_draw=self.config.gestures.air_draw,
            two_hand=self.config.gestures.two_hand,
            recognition=self.config.recognition,
            gesture_events=self.config.gesture_events,
            gesture_mapping=self.config.gesture_mapping,
        )

    def _extractor_for(self, hand: HandLandmarks, index: int) -> FeatureExtractor:
        key = hand.handedness or f"hand_{index}"
        if key not in self._multi_extractors:
            self._multi_extractors[key] = FeatureExtractor()
        return self._multi_extractors[key]

    def run_camera(self, max_frames: int | None = None) -> None:
        settings = CameraSettings(
            index=self.config.camera.index,
            width=self.config.camera.width,
            height=self.config.camera.height,
            fps=self.config.camera.fps,
        )
        frame_count = 0
        with OpenCVCamera(settings) as camera, MediaPipeHandTracker() as tracker:
            while max_frames is None or frame_count < max_frames:
                frame = camera.read()
                if frame is None:
                    continue
                frame_count += 1
                hands = tracker.track(frame)
                if not hands:
                    self.reset()
                    continue
                self.process_hands(hands)

    def _dispatch(self, event: GestureEvent) -> None:
        for callback in [*self._callbacks.get(event.gesture, []), *self._callbacks.get("*", [])]:
            callback(event)
        for broker in self._brokers:
            if getattr(broker, "handles_features", False):
                continue
            publish = getattr(broker, "publish", None)
            if publish is None:
                background_publish = getattr(broker, "publish_background", None)
                if background_publish is not None:
                    background_publish(event)
                continue
            result = publish(event)
            if inspect.isawaitable(result):
                self._schedule(result)

    def _schedule(self, awaitable) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(awaitable)
            return
        loop.create_task(awaitable)
