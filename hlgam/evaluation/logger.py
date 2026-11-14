from __future__ import annotations

from pathlib import Path

from hlgam.models import GestureEvent


class GestureEventLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: GestureEvent) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(event.to_json())
            file.write("\n")
