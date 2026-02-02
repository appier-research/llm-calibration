"""Training pipeline for confidence probes."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

from .base import BaseProbe

logger = logging.getLogger(__name__)


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
    loss_type: Literal["bce", "mse", "ce"] = "bce"
    optimizer_type: Literal["adamw", "sgd", "lbfgs"] = "sgd"
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
        if config.optimizer_type == "adamw":
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
        else:
            raise ValueError(f"Unknown loss type: {config.loss_type}")
        
        # Setup learning rate scheduler (skip for LBFGS which manages its own step size)
        self.scheduler = None
        if config.optimizer_type != "lbfgs" and config.lr_scheduler_type != "none":
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
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0

    def train(
        self,
        train_dataset: ProbeDataset,
        val_dataset: ProbeDataset,
    ) -> dict:
        """
        Train the probe.
        
        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
        
        Returns:
            Dict with training history and final metrics.
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        
        logger.info(f"Training on {len(train_dataset)} examples, validating on {len(val_dataset)}")
        
        history = {
            "train_loss": [],
            "val_loss": [],
        }
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_loss = self._train_epoch(train_loader)
            
            # Validation
            val_loss = self._validate(val_loader)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log to wandb
            if self.use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                })
            
            # Check for improvement
            if val_loss < self.best_val_loss:
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
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Step learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Load best model
        if self.output_dir and (self.output_dir / "best_probe.pt").exists():
            best_checkpoint = torch.load(self.output_dir / "best_probe.pt", weights_only=False)
            self.probe.load_state_dict(best_checkpoint["state_dict"])
        
        # Final metrics
        final_metrics = {
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "total_epochs": len(history["train_loss"]),
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
                },
            }, indent=4))
        
        return {
            "history": history,
            "metrics": final_metrics,
        }

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

    def _validate(self, loader: DataLoader) -> float:
        """Validate and return average loss."""
        self.probe.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for hidden_states, targets in loader:
                hidden_states = hidden_states.to(self.device)
                targets = targets.to(self.device)
                
                # For CE loss, convert continuous targets to class indices
                if self.config.loss_type == "ce":
                    targets = (targets * (self.config.num_classes - 1)).round().long()
                
                predictions = self.probe(hidden_states)
                loss = self.loss_fn(predictions, targets)
                
                total_loss += loss.item() * len(targets)
        
        return total_loss / len(loader.dataset)

    def evaluate(self, dataset: ProbeDataset) -> dict:
        """
        Evaluate probe on a dataset.
        
        Returns various metrics comparing predictions to targets.
        """
        from src.metrics.custom import c_star_metrics
        
        self.probe.eval()
        
        loader = DataLoader(dataset, batch_size=self.config.batch_size)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for hidden_states, targets in loader:
                hidden_states = hidden_states.to(self.device)
                
                # For classifier probes (CE loss), use predict_confidence() to get scalar outputs
                if self.config.loss_type == "ce":
                    predictions = self.probe.predict_confidence(hidden_states)
                else:
                    predictions = self.probe(hidden_states)
                    # For BCE probes with raw logits, apply sigmoid
                    if not self.config.apply_sigmoid:
                        predictions = torch.sigmoid(predictions)
                
                all_predictions.extend(predictions.cpu().tolist())
                all_targets.extend(targets.tolist())
        
        # Compute metrics
        metrics = c_star_metrics(all_predictions, all_targets)
        
        return {
            "mae": metrics.mae,
            "mse": metrics.mse,
            "rmse": metrics.rmse,
            "pearson_r": metrics.pearson_r,
            "spearman_r": metrics.spearman_r,
            "ece": metrics.ece,
        }
