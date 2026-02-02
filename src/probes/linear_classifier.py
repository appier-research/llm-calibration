"""Linear classifier probe for confidence estimation via classification."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseProbe


class LinearClassifierProbe(BaseProbe):
    """
    Linear classifier probe: hidden_states -> k+1 class logits.
    
    Architecture: Linear(hidden_dim, num_classes)
    
    Training: Uses cross-entropy loss with discretized targets.
    Inference: Converts class probabilities to expected value in [0, 1].
    
    For k samples used to estimate ground truth accuracy:
    - num_classes = k + 1 (classes 0, 1, ..., k)
    - Target mapping: class_idx = round(target * k)
    - Prediction: sum(softmax(logits) * [0/k, 1/k, ..., k/k])
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        bias: bool = True,
    ):
        """
        Args:
            input_dim: Dimension of input hidden states.
            num_classes: Number of classes (k + 1 where k is the number of samples).
            bias: Whether to include bias term.
        """
        super().__init__(input_dim)
        
        self.num_classes = num_classes
        self.linear = nn.Linear(input_dim, num_classes, bias=bias)
        
        # Register class values as buffer for inference
        # Values are [0/k, 1/k, ..., k/k] = [0, 1/k, ..., 1]
        class_values = torch.linspace(0, 1, num_classes)
        self.register_buffer("class_values", class_values)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute class logits from hidden states.
        
        Args:
            hidden_states: Shape (batch, hidden_dim)
        
        Returns:
            Logits of shape (batch, num_classes)
        """
        hidden_states = self.normalize(hidden_states)
        logits = self.linear(hidden_states)  # (batch, num_classes)
        return logits

    def predict_confidence(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict confidence as expected value from class probabilities.
        
        Args:
            hidden_states: Shape (batch, hidden_dim)
        
        Returns:
            Confidence scores of shape (batch,) in [0, 1].
        """
        logits = self.forward(hidden_states)  # (batch, num_classes)
        probs = F.softmax(logits, dim=-1)  # (batch, num_classes)
        # Expected value: sum(p_i * v_i) where v_i = i/k
        confidence = (probs * self.class_values).sum(dim=-1)  # (batch,)
        return confidence

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict confidence scores (inference mode)."""
        self.eval()
        with torch.no_grad():
            return self.predict_confidence(hidden_states)

    def get_weights(self) -> torch.Tensor:
        """Get the linear layer weights for interpretability."""
        return self.linear.weight.data

    def save(self, path: str | Path) -> None:
        """Save probe weights and config to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "class_name": self.__class__.__name__,
            "config": {
                "num_classes": self.num_classes,
            },
        }, path)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "LinearClassifierProbe":
        """Load probe weights from file."""
        checkpoint = torch.load(path, weights_only=False)
        
        config = checkpoint.get("config", {})
        num_classes = config.get("num_classes", kwargs.get("num_classes", 11))
        
        # Create instance
        probe = cls(
            input_dim=checkpoint["input_dim"],
            num_classes=num_classes,
        )
        probe.load_state_dict(checkpoint["state_dict"])
        
        return probe
