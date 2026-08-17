"""Training pipeline for confidence probes."""

import json
import logging
import math
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

from .base import BaseProbe

logger = logging.getLogger(__name__)

CHECKPOINT_METRICS = (
    "val_loss",
    "mae",
    "mse",
    "rmse",
    "pearson_r",
    "spearman_r",
    "ece",
)
CHECKPOINT_METRIC_DIRECTIONS = {
    "val_loss": "min",
    "mae": "min",
    "mse": "min",
    "rmse": "min",
    "pearson_r": "max",
    "spearman_r": "max",
    "ece": "min",
}


def ranknet_logistic_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: Literal["mean", "sum"] = "mean",
    temperature: float = 1.0,
    tie_loss_weight: float = 0.0,
) -> torch.Tensor:
    """Compute RankNet's pairwise logistic loss for scalar probe scores."""
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("RankNet temperature must be finite and positive.")
    if not math.isfinite(tie_loss_weight) or tie_loss_weight < 0.0:
        raise ValueError("RankNet tie loss weight must be finite and non-negative.")
    if reduction not in ("mean", "sum"):
        raise ValueError(f"Unknown reduction: {reduction}")
    if scores.ndim != 1 or targets.ndim != 1 or scores.shape != targets.shape:
        raise ValueError(
            "RankNet scores and targets must be one-dimensional tensors with "
            f"matching shapes, got {tuple(scores.shape)} and {tuple(targets.shape)}."
        )

    score_differences = scores[:, None] - scores[None, :]
    target_differences = targets[:, None] - targets[None, :]
    upper_triangle = torch.triu(
        torch.ones_like(target_differences, dtype=torch.bool),
        diagonal=1,
    )
    comparable_pairs = upper_triangle & target_differences.ne(0)

    # Keep the zero connected to scores so all-tie batches remain
    # differentiable and produce zero gradients.
    loss = scores.sum() * 0.0
    if torch.any(comparable_pairs):
        directions = torch.sign(target_differences[comparable_pairs])
        signed_margins = (
            directions * score_differences[comparable_pairs] / temperature
        )
        pair_losses = F.softplus(-signed_margins)
        loss = pair_losses.sum() if reduction == "sum" else pair_losses.mean()

    tied_pairs = upper_triangle & target_differences.eq(0)
    if tie_loss_weight > 0.0 and torch.any(tied_pairs):
        tie_margins = score_differences[tied_pairs] / temperature
        # Neutral (0.5) preference is minimized at equal scores. Subtract the
        # constant minimum so this auxiliary term is zero there.
        tie_losses = (
            0.5 * (F.softplus(-tie_margins) + F.softplus(tie_margins))
            - math.log(2.0)
        )
        tie_loss = tie_losses.sum() if reduction == "sum" else tie_losses.mean()
        loss = loss + tie_loss_weight * tie_loss

    return loss


class ProbeDataset(Dataset):
    """
    Dataset for probe training.
    
    Each item is (hidden_state, c_star) where:
    - hidden_state: LLM hidden representation for question x
    - c_star: Ground truth accuracy from sampling (target)
    """

    def __init__(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        Args:
            hidden_states: Tensor of shape (n_examples, hidden_dim).
            targets: Tensor of shape (n_examples,) with c*(x) values.
        """
        self.hidden_states = hidden_states
        self.targets = targets
        
        assert len(hidden_states) == len(targets), "Mismatched lengths"

    def __len__(self) -> int:
        return len(self.hidden_states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hidden_states[idx], self.targets[idx]

    @classmethod
    def from_safetensors(
        cls,
        data_dir: str | Path,
        layer_idx: int,
    ) -> "ProbeDataset":
        """
        Load dataset from safetensors files.
        
        Args:
            data_dir: Directory containing hidden_states.safetensors and targets.safetensors.
            layer_idx: Which layer's activations to use (0 to n_layers-1).
        
        Returns:
            ProbeDataset for the specified layer.
        """
        data_dir = Path(data_dir)
        
        # Load hidden states: shape (n_examples, n_layers, hidden_dim)
        hs_data = load_file(data_dir / "hidden_states.safetensors")
        hidden_states = hs_data["hidden_states"]
        
        # Slice to get specific layer: (n_examples, hidden_dim)
        hidden_states = hidden_states[:, layer_idx, :].float()
        
        # Load targets: shape (n_examples,)
        targets_data = load_file(data_dir / "targets.safetensors")
        targets = targets_data["targets"]
        
        return cls(hidden_states, targets)

    @classmethod
    def get_num_layers(cls, data_dir: str | Path) -> int:
        """Get the number of layers from metadata."""
        data_dir = Path(data_dir)
        with open(data_dir / "metadata.json") as f:
            metadata = json.load(f)
        return metadata["n_layers"]

    @classmethod
    def get_hidden_dim(cls, data_dir: str | Path) -> int:
        """Get the hidden dimension from metadata."""
        data_dir = Path(data_dir)
        with open(data_dir / "metadata.json") as f:
            metadata = json.load(f)
        return metadata["hidden_dim"]

    @classmethod
    def get_num_samples(cls, data_dir: str | Path) -> int:
        """Get the number of samples from the targets file."""
        data_dir = Path(data_dir)
        targets_data = load_file(data_dir / "targets.safetensors")
        return targets_data["targets"].shape[0]

    @classmethod
    def from_safetensors_pooled(
        cls,
        data_dir: str | Path,
        pooling: Literal["mean", "max"],
    ) -> "ProbeDataset":
        """
        Load dataset with hidden states pooled across all layers.
        
        Args:
            data_dir: Directory containing hidden_states.safetensors and targets.safetensors.
            pooling: Pooling strategy ("mean" or "max").
        
        Returns:
            ProbeDataset with pooled hidden states.
        """
        data_dir = Path(data_dir)
        
        # Load hidden states: shape (n_examples, n_layers, hidden_dim)
        hs_data = load_file(data_dir / "hidden_states.safetensors")
        hidden_states = hs_data["hidden_states"]
        
        # Pool across layers: (n_examples, hidden_dim)
        if pooling == "mean":
            hidden_states = hidden_states.mean(dim=1).float()
        elif pooling == "max":
            hidden_states = hidden_states.max(dim=1).values.float()
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        # Load targets: shape (n_examples,)
        targets_data = load_file(data_dir / "targets.safetensors")
        targets = targets_data["targets"]
        
        return cls(hidden_states, targets)

@dataclass
class TrainingConfig:
    """Configuration for probe training."""
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10
    device: str = "cuda"
    loss_type: Literal["bce", "mse", "ce", "ranknet_logistic_loss"] = "bce"
    ranknet_logistic_loss_temperature: float = 1.0
    ranknet_tie_loss_weight: float = 0.0
    checkpoint_metric: Literal[
        "val_loss", "mae", "mse", "rmse", "pearson_r", "spearman_r", "ece"
    ] = "val_loss"
    checkpoint_metric_mode: Literal["auto", "min", "max"] = "auto"
    optimizer_type: Literal["adamw", "sgd", "lbfgs", "closed_form"] = "sgd"
    lbfgs_max_iter: int = 20  # Max iterations per LBFGS step
    lbfgs_history_size: int = 100
    apply_sigmoid: bool = True  # Whether probe applies sigmoid (False -> use BCEWithLogitsLoss for BCE)
    lr_scheduler_type: Literal["none", "linear", "cosine"] = "none"
    num_classes: Optional[int] = None  # Number of classes for CE loss (k+1)

class ProbeTrainer:
    """
    Trainer for confidence probes.
    
    Handles:
    - Training loop with early stopping
    - Checkpointing best model
    - Logging metrics (optionally to wandb)
    """

    def __init__(
        self,
        probe: BaseProbe,
        config: TrainingConfig,
        output_dir: Optional[str | Path] = None,
        use_wandb: bool = False,
    ):
        """
        Args:
            probe: The probe to train.
            config: Training configuration.
            output_dir: Directory for saving checkpoints and logs.
            use_wandb: Whether to log to wandb (assumes wandb.init already called).
        """
        self.probe = probe
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else None
        self.use_wandb = use_wandb
        
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.probe.to(self.device)
        
        # Setup optimizer
        if config.optimizer_type == "closed_form":
            if config.loss_type != "mse":
                raise ValueError("closed_form optimizer requires loss_type='mse'")
            if config.apply_sigmoid:
                raise ValueError("closed_form optimizer requires apply_sigmoid=False")
            self.optimizer = None
        elif config.optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                probe.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(
                probe.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=0.9,
            )
        elif config.optimizer_type == "lbfgs":
            self.optimizer = torch.optim.LBFGS(
                probe.parameters(),
                lr=config.learning_rate,
                max_iter=config.lbfgs_max_iter,
                history_size=config.lbfgs_history_size,
                line_search_fn="strong_wolfe",
            )
        else:
            raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")
        
        # Setup loss function
        if config.loss_type == "bce":
            if config.apply_sigmoid:
                self.loss_fn = nn.BCELoss()
            else:
                # Use BCEWithLogitsLoss when probe outputs raw logits (more numerically stable)
                self.loss_fn = nn.BCEWithLogitsLoss()
        elif config.loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif config.loss_type == "ce":
            # Cross-entropy loss for classifier probes
            if config.num_classes is None:
                raise ValueError("num_classes must be specified for CE loss")
            self.loss_fn = nn.CrossEntropyLoss()
        elif config.loss_type == "ranknet_logistic_loss":
            if config.apply_sigmoid:
                raise ValueError(
                    "ranknet_logistic_loss requires raw probe scores; set "
                    "apply_sigmoid=False (CLI: --no_apply_sigmoid)"
                )
            if (
                not math.isfinite(config.ranknet_logistic_loss_temperature)
                or config.ranknet_logistic_loss_temperature <= 0.0
            ):
                raise ValueError("RankNet temperature must be finite and positive.")
            if (
                not math.isfinite(config.ranknet_tie_loss_weight)
                or config.ranknet_tie_loss_weight < 0.0
            ):
                raise ValueError(
                    "RankNet tie loss weight must be finite and non-negative."
                )
            self.loss_fn = partial(
                ranknet_logistic_loss,
                temperature=config.ranknet_logistic_loss_temperature,
                tie_loss_weight=config.ranknet_tie_loss_weight,
            )
        else:
            raise ValueError(f"Unknown loss type: {config.loss_type}")
        
        # Setup learning rate scheduler (skip for LBFGS and closed_form)
        self.scheduler = None
        if config.optimizer_type not in ("lbfgs", "closed_form") and config.lr_scheduler_type != "none":
            if config.lr_scheduler_type == "linear":
                self.scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=0.0,
                    total_iters=config.num_epochs,
                )
            elif config.lr_scheduler_type == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=config.num_epochs,
                )
        
        # Best model tracking
        if config.checkpoint_metric not in CHECKPOINT_METRICS:
            raise ValueError(
                f"Unknown checkpoint metric: {config.checkpoint_metric}. "
                f"Choose from {CHECKPOINT_METRICS}."
            )
        if config.checkpoint_metric_mode not in ("auto", "min", "max"):
            raise ValueError(
                f"Unknown checkpoint metric mode: {config.checkpoint_metric_mode}"
            )
        self.checkpoint_metric = config.checkpoint_metric
        self.checkpoint_metric_mode = (
            CHECKPOINT_METRIC_DIRECTIONS[self.checkpoint_metric]
            if config.checkpoint_metric_mode == "auto"
            else config.checkpoint_metric_mode
        )
        self.best_checkpoint_metric_value = (
            float("inf") if self.checkpoint_metric_mode == "min" else float("-inf")
        )
        self.best_val_metrics: dict[str, float] | None = None
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0

    def _is_checkpoint_improvement(self, value: float) -> bool:
        """Return whether a validation metric improves on the current best."""
        if not math.isfinite(value):
            return False
        if self.checkpoint_metric_mode == "min":
            return value < self.best_checkpoint_metric_value
        return value > self.best_checkpoint_metric_value

    def train(
        self,
        train_dataset: ProbeDataset,
        val_dataset: ProbeDataset | None,
    ) -> dict:
        """
        Train the probe.
        
        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset. If None, train for the configured
                number of epochs and save the final checkpoint as best_probe.pt.
        
        Returns:
            Dict with training history and final metrics.
        """
        if self.config.optimizer_type == "closed_form":
            if val_dataset is None:
                raise ValueError("closed_form training requires a validation dataset")
            return self._train_closed_form(train_dataset, val_dataset)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )
        
        if val_dataset is None:
            logger.info(f"Training on {len(train_dataset)} examples with no validation")
        else:
            logger.info(f"Training on {len(train_dataset)} examples, validating on {len(val_dataset)}")
        
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": [],
        }
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_loss = self._train_epoch(train_loader)
            
            # Validation
            val_metrics = None if val_loader is None else self._validate(val_loader)
            val_loss = None if val_metrics is None else val_metrics["val_loss"]
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(val_metrics)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log to wandb
            if self.use_wandb:
                import wandb
                log_data = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "learning_rate": current_lr,
                }
                if val_metrics is not None:
                    log_data["val_loss"] = val_loss
                    log_data.update({
                        f"val/{name}": value
                        for name, value in val_metrics.items()
                        if name != "val_loss"
                    })
                    log_data["checkpoint_metric_value"] = val_metrics[
                        self.checkpoint_metric
                    ]
                wandb.log(log_data)
            
            # Check for improvement
            if val_metrics is None:
                self.best_val_loss = train_loss
                self.best_epoch = epoch
            else:
                checkpoint_value = val_metrics[self.checkpoint_metric]
                if self._is_checkpoint_improvement(checkpoint_value):
                    self.best_checkpoint_metric_value = checkpoint_value
                    self.best_val_metrics = dict(val_metrics)
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0

                    # Save best model
                    if self.output_dir:
                        self.probe.save(self.output_dir / "best_probe.pt")
                else:
                    self.patience_counter += 1
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                if val_loss is None:
                    logger.info(
                        f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                        f"Train Loss: {train_loss:.4f}"
                    )
                else:
                    checkpoint_value = val_metrics[self.checkpoint_metric]
                    logger.info(
                        f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                        f"{self.checkpoint_metric}: {checkpoint_value:.4f}"
                    )
            
            # Early stopping
            if val_loss is not None and self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Step learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Load best model or save the fixed-epoch final checkpoint.
        if val_dataset is None and self.output_dir:
            self.probe.save(self.output_dir / "best_probe.pt")
        elif self.output_dir and (self.output_dir / "best_probe.pt").exists():
            best_checkpoint = torch.load(self.output_dir / "best_probe.pt", weights_only=False)
            self.probe.load_state_dict(best_checkpoint["state_dict"])
        
        # Final metrics
        final_metrics = {
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "total_epochs": len(history["train_loss"]),
            "checkpoint_metric": self.checkpoint_metric,
            "checkpoint_metric_mode": self.checkpoint_metric_mode,
            "best_checkpoint_metric_value": (
                None
                if val_dataset is None
                else self.best_checkpoint_metric_value
            ),
            "best_val_metrics": self.best_val_metrics,
        }
        
        # Save training history
        if self.output_dir:
            history_path = self.output_dir / "training_history.json"
            history_path.write_text(json.dumps({
                "history": history,
                "metrics": final_metrics,
                "config": {
                    "learning_rate": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                    "batch_size": self.config.batch_size,
                    "num_epochs": self.config.num_epochs,
                    "early_stopping_patience": self.config.early_stopping_patience,
                    "device": self.config.device,
                    "loss_type": self.config.loss_type,
                    "optimizer_type": self.config.optimizer_type,
                    "lr_scheduler_type": self.config.lr_scheduler_type,
                    "ranknet_logistic_loss_temperature": self.config.ranknet_logistic_loss_temperature,
                    "ranknet_tie_loss_weight": self.config.ranknet_tie_loss_weight,
                    "checkpoint_metric": self.config.checkpoint_metric,
                    "checkpoint_metric_mode": self.config.checkpoint_metric_mode,
                },
            }, indent=4))
        
        return {
            "history": history,
            "metrics": final_metrics,
        }

    def _fit_closed_form(self, train_dataset: "ProbeDataset") -> float:
        """Fit probe weights using closed-form ridge regression.

        Solves: min ||Xw + b - y||² + λ||w||²
        via the normal equations with centered data (bias is not regularized).

        Returns training MSE.
        """
        X = train_dataset.hidden_states.to(self.device)
        y = train_dataset.targets.to(self.device)

        X = self.probe.normalize(X)

        X_mean = X.mean(dim=0)
        y_mean = y.mean()
        X_c = X - X_mean
        y_c = y - y_mean

        n, d = X_c.shape
        A = X_c.T @ X_c + (self.config.weight_decay * n) * torch.eye(d, device=self.device, dtype=X.dtype)
        w = torch.linalg.solve(A, X_c.T @ y_c)

        bias = y_mean - X_mean @ w

        with torch.no_grad():
            self.probe.linear.weight.copy_(w.unsqueeze(0))
            self.probe.linear.bias.copy_(bias.unsqueeze(0))

        with torch.no_grad():
            preds = self.probe(train_dataset.hidden_states.to(self.device))
            train_loss = nn.functional.mse_loss(preds, y).item()

        return train_loss

    def _train_closed_form(
        self,
        train_dataset: "ProbeDataset",
        val_dataset: "ProbeDataset",
    ) -> dict:
        """Train using closed-form ridge regression and return results."""
        train_loss = self._fit_closed_form(train_dataset)

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        val_metrics = self._validate(val_loader)
        val_loss = val_metrics["val_loss"]
        checkpoint_value = val_metrics[self.checkpoint_metric]

        self.best_val_loss = val_loss
        self.best_val_metrics = dict(val_metrics)
        self.best_checkpoint_metric_value = checkpoint_value
        self.best_epoch = 0

        if self.output_dir:
            self.probe.save(self.output_dir / "best_probe.pt")

        if self.use_wandb:
            import wandb
            log_data = {
                "epoch": 0,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "checkpoint_metric_value": checkpoint_value,
            }
            log_data.update({
                f"val/{name}": value
                for name, value in val_metrics.items()
                if name != "val_loss"
            })
            wandb.log(log_data)

        logger.info(
            f"Closed-form solve - Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"{self.checkpoint_metric}: {checkpoint_value:.4f}"
        )

        history = {
            "train_loss": [train_loss],
            "val_loss": [val_loss],
            "val_metrics": [val_metrics],
        }
        final_metrics = {
            "best_val_loss": val_loss,
            "best_epoch": 0,
            "total_epochs": 1,
            "checkpoint_metric": self.checkpoint_metric,
            "checkpoint_metric_mode": self.checkpoint_metric_mode,
            "best_checkpoint_metric_value": checkpoint_value,
            "best_val_metrics": val_metrics,
        }

        if self.output_dir:
            history_path = self.output_dir / "training_history.json"
            history_path.write_text(json.dumps({
                "history": history,
                "metrics": final_metrics,
                "config": {
                    "learning_rate": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                    "batch_size": self.config.batch_size,
                    "num_epochs": self.config.num_epochs,
                    "early_stopping_patience": self.config.early_stopping_patience,
                    "device": self.config.device,
                    "loss_type": self.config.loss_type,
                    "optimizer_type": self.config.optimizer_type,
                    "lr_scheduler_type": self.config.lr_scheduler_type,
                    "ranknet_logistic_loss_temperature": self.config.ranknet_logistic_loss_temperature,
                    "ranknet_tie_loss_weight": self.config.ranknet_tie_loss_weight,
                    "checkpoint_metric": self.config.checkpoint_metric,
                    "checkpoint_metric_mode": self.config.checkpoint_metric_mode,
                },
            }, indent=4))

        return {"history": history, "metrics": final_metrics}

    def _train_epoch(self, loader: DataLoader) -> float:
        """Train for one epoch."""
        self.probe.train()
        total_loss = 0.0
        
        if self.config.optimizer_type == "lbfgs":
            # LBFGS uses full-batch and requires a closure
            return self._train_epoch_lbfgs(loader)
        
        for hidden_states, targets in loader:
            hidden_states = hidden_states.to(self.device)
            targets = targets.to(self.device)
            
            # For CE loss, convert continuous targets to class indices
            if self.config.loss_type == "ce":
                # Map [0, 1] to class indices [0, num_classes-1]
                targets = (targets * (self.config.num_classes - 1)).round().long()
            
            # Forward pass
            predictions = self.probe(hidden_states)
            loss = self.loss_fn(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * len(targets)
        
        return total_loss / len(loader.dataset)

    def _train_epoch_lbfgs(self, loader: DataLoader) -> float:
        """Train for one epoch using LBFGS (full-batch)."""
        # Collect all data for full-batch optimization
        all_hidden = []
        all_targets = []
        for hidden_states, targets in loader:
            all_hidden.append(hidden_states.to(self.device))
            all_targets.append(targets.to(self.device))
        
        hidden_states = torch.cat(all_hidden, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # For CE loss, convert continuous targets to class indices
        if self.config.loss_type == "ce":
            targets = (targets * (self.config.num_classes - 1)).round().long()
        
        # LBFGS requires a closure that recomputes the loss
        loss_value = 0.0
        
        def closure():
            nonlocal loss_value
            self.optimizer.zero_grad()
            predictions = self.probe(hidden_states)
            loss = self.loss_fn(predictions, targets)
            
            # Manual L2 regularization (LBFGS doesn't support weight_decay)
            if self.config.weight_decay > 0:
                l2_reg = sum(p.pow(2).sum() for p in self.probe.parameters())
                loss = loss + 0.5 * self.config.weight_decay * l2_reg
            
            loss.backward()
            loss_value = loss.item()
            return loss
        
        self.optimizer.step(closure)
        
        return loss_value

    def _outputs_to_confidences(self, outputs: torch.Tensor) -> torch.Tensor:
        """Convert probe training outputs to scalar confidence predictions."""
        if self.config.loss_type == "ce":
            probabilities = F.softmax(outputs, dim=-1)
            return (probabilities * self.probe.class_values).sum(dim=-1)
        if (
            not self.config.apply_sigmoid
            and self.config.optimizer_type != "closed_form"
        ):
            return torch.sigmoid(outputs)
        return outputs

    def _validate(self, loader: DataLoader) -> dict[str, float]:
        """Validate once and return loss plus confidence-quality metrics."""
        from src.metrics.custom import c_star_metrics

        self.probe.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for hidden_states, targets in loader:
                hidden_states = hidden_states.to(self.device)
                targets = targets.to(self.device)
                loss_targets = targets
                
                # For CE loss, convert continuous targets to class indices.
                if self.config.loss_type == "ce":
                    loss_targets = (
                        targets * (self.config.num_classes - 1)
                    ).round().long()
                
                outputs = self.probe(hidden_states)
                loss = self.loss_fn(outputs, loss_targets)
                confidences = self._outputs_to_confidences(outputs)
                
                total_loss += loss.item() * len(targets)
                all_predictions.extend(confidences.cpu().tolist())
                all_targets.extend(targets.cpu().tolist())
        
        metrics = c_star_metrics(all_predictions, all_targets)
        return {
            "val_loss": total_loss / len(loader.dataset),
            "mae": metrics.mae,
            "mse": metrics.mse,
            "rmse": metrics.rmse,
            "pearson_r": metrics.pearson_r,
            "spearman_r": metrics.spearman_r,
            "ece": metrics.ece,
        }

    def evaluate(self, dataset: ProbeDataset) -> dict[str, float]:
        """Evaluate confidence-quality metrics on a dataset."""
        loader = DataLoader(dataset, batch_size=self.config.batch_size)
        validation_metrics = self._validate(loader)
        return {
            name: value
            for name, value in validation_metrics.items()
            if name != "val_loss"
        }
