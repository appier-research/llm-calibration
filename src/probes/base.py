"""Base probe interface for learned confidence estimation."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class BaseProbe(ABC, nn.Module):
    """
    Abstract base class for confidence probes.
    
    Probes learn to predict c*(x) from hidden states.
    They map hidden representations to confidence scores.
    """

    def __init__(self, input_dim: int):
        """
        Args:
            input_dim: Dimension of input hidden states.
        """
        super().__init__()
        self.input_dim = input_dim
        
        # Normalization stats (set via set_normalization)
        self.register_buffer("input_mean", None)
        self.register_buffer("input_std", None)

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set normalization stats computed from training data."""
        self.input_mean = mean
        self.input_std = std

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization if stats are set."""
        if self.input_mean is not None:
            return (x - self.input_mean) / self.input_std
        return x

    @abstractmethod
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict confidence from hidden states.
        
        Args:
            hidden_states: Input tensor of shape (batch, hidden_dim).
        
        Returns:
            Confidence scores of shape (batch,) in [0, 1].
        """
        pass

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict confidence scores (inference mode)."""
        self.eval()
        with torch.no_grad():
            return self.forward(hidden_states)

    def save(self, path: str | Path) -> None:
        """Save probe weights to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "class_name": self.__class__.__name__,
        }, path)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "BaseProbe":
        """Load probe weights from file."""
        checkpoint = torch.load(path)
        
        # Create instance
        probe = cls(input_dim=checkpoint["input_dim"], **kwargs)
        probe.load_state_dict(checkpoint["state_dict"])
        
        return probe

