from __future__ import annotations

import time
from dataclasses import dataclass

from hlgam.models import Frame


@dataclass(frozen=True)
class CameraSettings:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True


class OpenCVCamera:
    def __init__(self, settings: CameraSettings | None = None) -> None:
        self.settings = settings or CameraSettings()
        self._cv2 = None
        self._capture = None
        self._frame_index = 0

    def open(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV camera capture requires the opencv-python optional dependency") from exc

        capture = cv2.VideoCapture(self.settings.index)
        if not capture.isOpened():
            raise RuntimeError(f"unable to open camera index {self.settings.index}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.height)
        capture.set(cv2.CAP_PROP_FPS, self.settings.fps)
        self._cv2 = cv2
        self._capture = capture

    def read(self) -> Frame | None:
        if self._capture is None:
            self.open()
        assert self._capture is not None
        ok, image = self._capture.read()
        if not ok:
            return None
        if self.settings.mirror:
            assert self._cv2 is not None
            image = self._cv2.flip(image, 1)
        self._frame_index += 1
        return Frame(image=image, timestamp=time.perf_counter(), index=self._frame_index)

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None

    def __enter__(self) -> OpenCVCamera:
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
