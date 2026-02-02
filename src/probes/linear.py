"""Linear probe for confidence estimation."""

import torch
import torch.nn as nn

from .base import BaseProbe


class LinearProbe(BaseProbe):
    """
    Simple linear probe: hidden_states -> confidence.
    
    Architecture: Linear(hidden_dim, 1) + Sigmoid
    
    Fast to train and provides a strong baseline.
    """

    def __init__(
        self,
        input_dim: int,
        bias: bool = True,
        apply_sigmoid: bool = True,
    ):
        """
        Args:
            input_dim: Dimension of input hidden states.
            bias: Whether to include bias term.
            apply_sigmoid: Whether to apply sigmoid to output (set False when using BCEWithLogitsLoss).
        """
        super().__init__(input_dim)
        
        self.apply_sigmoid = apply_sigmoid
        self.linear = nn.Linear(input_dim, 1, bias=bias)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict confidence from hidden states.
        
        Args:
            hidden_states: Shape (batch, hidden_dim)
        
        Returns:
            Confidence scores of shape (batch,) in [0, 1] if apply_sigmoid=True,
            otherwise raw logits.
        """
        hidden_states = self.normalize(hidden_states)
        logits = self.linear(hidden_states)  # (batch, 1)
        if self.apply_sigmoid:
            output = self.sigmoid(logits)  # (batch, 1)
        else:
            output = logits
        return output.squeeze(-1)  # (batch,)

    def get_weights(self) -> torch.Tensor:
        """Get the linear layer weights for interpretability."""
        return self.linear.weight.data.squeeze()

