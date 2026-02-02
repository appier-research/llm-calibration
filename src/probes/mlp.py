"""MLP probe for confidence estimation."""

import torch
import torch.nn as nn

from .base import BaseProbe


class MLPProbe(BaseProbe):
    """
    Multi-layer perceptron probe for confidence estimation.
    
    Architecture: 
        hidden_states -> Linear -> ReLU -> Dropout -> Linear -> Sigmoid
    
    More expressive than linear probe, can capture non-linear patterns.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
        apply_sigmoid: bool = True,
    ):
        """
        Args:
            input_dim: Dimension of input hidden states.
            hidden_dim: Dimension of hidden layers.
            num_layers: Number of hidden layers (minimum 1).
            dropout: Dropout probability.
            activation: Activation function ("relu", "gelu", "tanh").
            apply_sigmoid: Whether to apply sigmoid to output (set False when using BCEWithLogitsLoss).
        """
        super().__init__(input_dim)
        
        self.hidden_dim = hidden_dim
        self.num_layers = max(1, num_layers)
        self.dropout_prob = dropout
        self.activation_name = activation
        self.apply_sigmoid = apply_sigmoid
        
        # Build layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self._get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        if apply_sigmoid:
            layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation module by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.ReLU())

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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
        output = self.network(hidden_states)  # (batch, 1)
        return output.squeeze(-1)  # (batch,)

    @classmethod
    def load(cls, path, **kwargs) -> "MLPProbe":
        """Load MLP probe from checkpoint."""
        checkpoint = torch.load(path)
        
        # Extract config from checkpoint if available
        config = checkpoint.get("config", {})
        
        probe = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.1),
            activation=config.get("activation", "relu"),
            apply_sigmoid=config.get("apply_sigmoid", True),
        )
        probe.load_state_dict(checkpoint["state_dict"])
        
        return probe

    def save(self, path) -> None:
        """Save probe with config."""
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "class_name": self.__class__.__name__,
            "config": {
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout_prob,
                "activation": self.activation_name,
                "apply_sigmoid": self.apply_sigmoid,
            },
        }, path)

