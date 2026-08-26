"""Operational controls for pausing and resuming a scientific training run."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import RunConfig


@dataclass(frozen=True, slots=True)
class TrainingControl:
    """Controls that may change between exact epoch-boundary continuations."""

    stop_after_epoch: int | None = None
    rung: int = 0

    def validate(self, config: RunConfig) -> None:
        if self.rung < 0:
            raise ValueError("rung cannot be negative")
        if self.stop_after_epoch is not None:
            if self.stop_after_epoch <= 0:
                raise ValueError("stop_after_epoch must be positive")
            if self.stop_after_epoch > config.epochs:
                raise ValueError("stop_after_epoch cannot exceed configured epochs")

    def target_epoch(self, config: RunConfig) -> int:
        self.validate(config)
        return self.stop_after_epoch or config.epochs

    def to_dict(self) -> dict[str, int | None]:
        return asdict(self)
