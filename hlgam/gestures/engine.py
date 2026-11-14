from __future__ import annotations

from dataclasses import dataclass, field

from hlgam.gestures.air_draw import AirDrawConfig, AirDrawRecognizer
from hlgam.gestures.flick import FlickConfig, FlickRecognizer
from hlgam.gestures.motion import MotionGestureConfig, MotionGestureRecognizer
from hlgam.gestures.pinch_drag import PinchDragConfig, PinchDragRecognizer
from hlgam.gestures.pose import (
    FingerCountConfig,
    FingerCountRecognizer,
    FingerPointingConfig,
    FingerPointingRecognizer,
    FlatHandConfig,
    FlatHandRecognizer,
    StaticPoseConfig,
    StaticPoseRecognizer,
)
from hlgam.gestures.rotation import RotationConfig, RotationRecognizer
from hlgam.gestures.scroll import ScrollConfig, ScrollRecognizer
from hlgam.gestures.swipe import SwipeMotionConfig, SwipeMotionRecognizer
from hlgam.gestures.two_hand import TwoHandGestureConfig, TwoHandGestureRecognizer
from hlgam.gestures.zoom import ZoomConfig, ZoomRecognizer
from hlgam.ml.inference.motion import LearnedMotionRecognizer
from hlgam.ml.inference.static_pose import LearnedStaticPoseRecognizer
from hlgam.models import GestureEvent, GestureFeatures
from hlgam.recognition import RecognitionConfig


@dataclass(frozen=True)
class GestureEngineConfig:
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
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    gesture_events: dict[str, bool] = field(default_factory=dict)
    gesture_mapping: dict[str, str] = field(default_factory=dict)


class GestureEngine:
    def __init__(self, config: GestureEngineConfig | None = None) -> None:
        self.config = config or GestureEngineConfig()
        self.recognizers = self._create_single_hand_recognizers()
        self.learned_recognizers = self._create_learned_single_hand_recognizers()
        self._recognizers_by_hand: dict[str, list] = {}
        self._learned_recognizers_by_hand: dict[str, list] = {}
        self.two_hand_recognizer = TwoHandGestureRecognizer(self.config.two_hand)

    def _create_single_hand_recognizers(self) -> list:
        return [
            PinchDragRecognizer(self.config.pinch_drag),
            SwipeMotionRecognizer(self.config.swipe_motion),
            ZoomRecognizer(self.config.zoom),
            StaticPoseRecognizer(self.config.pose),
            FingerCountRecognizer(self.config.finger_count),
            FingerPointingRecognizer(self.config.finger_pointing),
            FlatHandRecognizer(self.config.flat_hand),
            FlickRecognizer(self.config.flick),
            RotationRecognizer(self.config.rotation),
            ScrollRecognizer(self.config.scroll),
            AirDrawRecognizer(self.config.air_draw),
            MotionGestureRecognizer(self.config.motion),
        ]

    def _create_learned_single_hand_recognizers(self) -> list:
        return [
            LearnedStaticPoseRecognizer(self.config.recognition.models.static_pose),
            LearnedMotionRecognizer(self.config.recognition.models.motion),
        ]

    def process(self, features: GestureFeatures) -> list[GestureEvent]:
        events: list[GestureEvent] = []
        if self.config.recognition.fsm.enabled:
            self._append_recognizer_events(events, self.recognizers, features)
        self._append_recognizer_events(events, self.learned_recognizers, features)
        return events

    def process_many(self, features_list: list[GestureFeatures]) -> list[GestureEvent]:
        if len(features_list) <= 1:
            return self.process(features_list[0]) if features_list else []

        events: list[GestureEvent] = []
        for index, features in enumerate(features_list):
            if self.config.recognition.fsm.enabled:
                self._append_recognizer_events(events, self._recognizers_for(features, index), features)
            self._append_recognizer_events(events, self._learned_recognizers_for(features, index), features)
        if self.config.recognition.fsm.enabled:
            for event in self.two_hand_recognizer.process_many(features_list):
                mapped_event = self._map_event(event)
                if mapped_event is not None:
                    events.append(mapped_event)
        return events

    def reset(self) -> None:
        for recognizer in self.recognizers:
            recognizer.reset()
        for recognizer in self.learned_recognizers:
            recognizer.reset()
        self._recognizers_by_hand.clear()
        self._learned_recognizers_by_hand.clear()
        self.two_hand_recognizer.reset()

    def _recognizers_for(self, features: GestureFeatures, index: int) -> list:
        key = f"{features.landmarks.handedness or 'hand'}_{index}"
        if key not in self._recognizers_by_hand:
            self._recognizers_by_hand[key] = self._create_single_hand_recognizers()
        return self._recognizers_by_hand[key]

    def _learned_recognizers_for(self, features: GestureFeatures, index: int) -> list:
        key = f"{features.landmarks.handedness or 'hand'}_{index}"
        if key not in self._learned_recognizers_by_hand:
            self._learned_recognizers_by_hand[key] = self._create_learned_single_hand_recognizers()
        return self._learned_recognizers_by_hand[key]

    def _append_recognizer_events(
        self,
        events: list[GestureEvent],
        recognizers: list,
        features: GestureFeatures,
    ) -> None:
        for recognizer in recognizers:
            for event in recognizer.process(features):
                mapped_event = self._map_event(event)
                if mapped_event is not None:
                    events.append(mapped_event)

    def _map_event(self, event: GestureEvent) -> GestureEvent | None:
        if not self.config.gesture_events.get(event.gesture, True):
            return None
        command = self.config.gesture_mapping.get(event.gesture, event.command)
        if command == event.command:
            return event
        return GestureEvent(
            gesture=event.gesture,
            phase=event.phase,
            confidence=event.confidence,
            timestamp=event.timestamp,
            position=event.position,
            velocity=event.velocity,
            command=command,
            metadata=event.metadata,
        )
