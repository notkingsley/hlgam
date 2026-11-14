from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FSMState(str, Enum):
    IDLE = "idle"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    COOLDOWN = "cooldown"


@dataclass
class FiniteStateMachine:
    state: FSMState = FSMState.IDLE
    entered_at: float = 0.0

    def transition(self, state: FSMState, timestamp: float) -> None:
        self.state = state
        self.entered_at = timestamp

    def elapsed(self, timestamp: float) -> float:
        return max(0.0, timestamp - self.entered_at)

    def reset(self) -> None:
        self.state = FSMState.IDLE
        self.entered_at = 0.0
