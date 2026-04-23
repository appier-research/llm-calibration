#!/usr/bin/env python
"""
Train confidence probes on hidden states, sweeping over all layers.

Each layer gets its own training run and wandb log.

Usage:
    uv run python scripts/train_probe.py \
        --train_dir outputs/triviaqa-train__Olmo-3-7B-Instruct__RTXPro6000/probing_dataset \
        --val_dir outputs/triviaqa-validation__Olmo-3-7B-Instruct__RTXPro6000/probing_dataset \
        --probe_type linear \
        --loss bce \
        --optimizer adamw \
        --lr 1e-3 \
        --weight_decay 1e-4 \
        --batch_size 32 \
        --epochs 100 \
        --patience 10 \
        --wandb_project probe-training \
        --output_dir outputs/probes/triviaqa__Olmo-3-7B-Instruct
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import wandb

from src.probes.linear import LinearProbe
from src.probes.linear_classifier import LinearClassifierProbe
from src.probes.mlp import MLPProbe
from src.probes.trainer import ProbeDataset, ProbeTrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train confidence probes, sweeping over all layers"
    )
    
    # Data paths
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Directory with training probing dataset (hidden_states.safetensors, targets.safetensors)",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default=None,
        help="Directory with validation probing dataset. Mutually exclusive with --val_num_examples.",
    )
    parser.add_argument(
        "--val_num_examples",
        type=int,
        default=None,
        help=(
            "If set, sample this many validation instances from the training set "
            "(using args.seed) instead of loading a separate --val_dir. Must be "
            "<= 0.5 * total training instances. Mutually exclusive with --val_dir."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save trained probes",
    )
    
    # Model configuration
    parser.add_argument(
        "--probe_type",
        type=str,
        choices=["linear", "mlp", "linear_classifier"],
        default="linear",
        help="Type of probe (default: linear)",
    )
    parser.add_argument(
        "--mlp_hidden_dim",
        type=int,
        default=16,
        help="Hidden dimension for MLP probe (default: 256)",
    )
    parser.add_argument(
        "--mlp_num_layers",
        type=int,
        default=1,
        help="Number of hidden layers for MLP probe (default: 2)",
    )
    parser.add_argument(
        "--mlp_dropout",
        type=float,
        default=0,
        help="Dropout for MLP probe (default: 0.1)",
    )
    
    # Training configuration
    parser.add_argument(
        "--loss",
        type=str,
        choices=["bce", "mse", "ce"],
        default="bce",
        help="Loss function (default: bce). Use 'ce' for linear_classifier probe.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "sgd", "lbfgs", "closed_form"],
        default="adamw",
        help="Optimizer (default: adamw). 'closed_form' uses ridge regression normal equations (requires --loss mse --no_apply_sigmoid --probe_type linear).",
    )
    parser.add_argument(
        "--lbfgs_max_iter",
        type=int,
        default=20,
        help="Max iterations per LBFGS step (default: 20)",
    )
    parser.add_argument(
        "--lbfgs_history_size",
        type=int,
        default=100,
        help="LBFGS history size (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of epochs (default: 100)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=["none", "linear", "cosine"],
        default="none",
        help="Learning rate scheduler (default: none)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for training (default: cuda)",
    )
    
    # Wandb configuration
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="probe-training",
        help="Wandb project name (default: probe-training)",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb entity/team (default: None, uses default)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    # Preprocessing
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Apply standardization to hidden states (z-score normalization)",
    )
    
    # Subsampling
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of training samples to use (default: None, uses all)",
    )
    
    # Layer selection
    parser.add_argument(
        "--layer_indices",
        type=str,
        default=None,
        help="Comma-separated layer indices to train (e.g., '22,23,24'). Default: train all layers.",
    )
    
    # Simulated k for target noise
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Simulate targets estimated with k samples (adds Binomial noise). Default: None (use original targets).",
    )
    
    # Sigmoid control
    parser.add_argument(
        "--apply_sigmoid",
        type=bool,
        default=True,
        help="Apply sigmoid to probe output (default: True). Use --no_apply_sigmoid to disable.",
    )
    parser.add_argument(
        "--no_apply_sigmoid",
        action="store_true",
        help="Do not apply sigmoid to probe output (use BCEWithLogitsLoss for BCE loss)",
    )
    
    # Linear classifier specific
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of classes for linear_classifier (k+1). Default: auto-detect from ground_truth_summary.json",
    )
    
    args = parser.parse_args()
    
    # Handle apply_sigmoid logic: default is True unless --no_apply_sigmoid is set
    if args.no_apply_sigmoid:
        args.apply_sigmoid = False
    
    # Validate --val_dir / --val_num_examples (exactly one must be provided)
    if args.val_dir is None and args.val_num_examples is None:
        parser.error("Must specify either --val_dir or --val_num_examples.")
    if args.val_dir is not None and args.val_num_examples is not None:
        parser.error("--val_dir and --val_num_examples are mutually exclusive.")
    if args.val_num_examples is not None and args.val_num_examples <= 0:
        parser.error("--val_num_examples must be a positive integer.")
    
    return args

def load_k_samples_from_summary(train_dir: Path) -> int:
    """
    Load k_samples from ground_truth_summary.json in the parent directory.
    
    Args:
        train_dir: Path to the probing_dataset directory.
    
    Returns:
        k_samples value from the summary file.
    """
    # train_dir is typically outputs/xxx/probing_dataset
    # ground_truth_summary.json is at outputs/xxx/ground_truth_summary.json
    summary_path = train_dir.parent / "ground_truth_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"ground_truth_summary.json not found at {summary_path}. "
            "Please specify --num_classes manually."
        )
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    return summary["k_samples"]


def simulate_k_targets(targets: torch.Tensor, k: int) -> torch.Tensor:
    """
    Simulate targets as if estimated from k samples instead of the original k.
    
    For each target p (true probability), we sample c ~ Binomial(k, p) and
    return c / k. This simulates the noise from using fewer samples.
    
    Args:
        targets: Original targets (probabilities in [0, 1]).
        k: Number of simulated samples.
    
    Returns:
        Noisy targets simulating k-sample estimation.
    """
    # Sample from Binomial(k, p) for each target p
    # torch.distributions.Binomial returns float, so we get exact counts
    binomial = torch.distributions.Binomial(total_count=k, probs=targets)
    counts = binomial.sample()
    return counts / k

def create_probe(probe_type: str, input_dim: int, args: argparse.Namespace):
    """Create a probe of the specified type."""
    if probe_type == "linear":
        return LinearProbe(input_dim=input_dim, apply_sigmoid=args.apply_sigmoid)
    elif probe_type == "mlp":
        return MLPProbe(
            input_dim=input_dim,
            hidden_dim=args.mlp_hidden_dim,
            num_layers=args.mlp_num_layers,
            dropout=args.mlp_dropout,
            apply_sigmoid=args.apply_sigmoid,
        )
    elif probe_type == "linear_classifier":
        return LinearClassifierProbe(
            input_dim=input_dim,
            num_classes=args.num_classes,
        )
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")


def train_layer(
    layer_idx: int,
    train_dir: Path,
    val_dir: Path | None,
    output_dir: Path,
    args: argparse.Namespace,
    hidden_dim: int,
    use_wandb: bool,
    subsample_indices: torch.Tensor | None = None,
    simulated_k: int | None = None,
    train_split_indices: torch.Tensor | None = None,
    val_split_indices: torch.Tensor | None = None,
) -> dict:
    """Train a probe for a single layer."""
    logger.info(f"Training probe for layer {layer_idx}")
    
    # Create layer-specific output directory
    layer_output_dir = output_dir / f"layer_{layer_idx:02d}"
    layer_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets for this layer
    train_dataset = ProbeDataset.from_safetensors(train_dir, layer_idx)
    if val_split_indices is not None:
        # Carve validation out of the training set using --val_num_examples
        val_dataset = ProbeDataset(
            train_dataset.hidden_states[val_split_indices],
            train_dataset.targets[val_split_indices],
        )
        train_dataset = ProbeDataset(
            train_dataset.hidden_states[train_split_indices],
            train_dataset.targets[train_split_indices],
        )
    else:
        val_dataset = ProbeDataset.from_safetensors(val_dir, layer_idx)
    
    # Subsample training set if requested (applied after val split)
    if subsample_indices is not None:
        train_dataset = ProbeDataset(
            train_dataset.hidden_states[subsample_indices],
            train_dataset.targets[subsample_indices],
        )
    
    # Apply simulated k to training targets (adds Binomial noise)
    if simulated_k is not None:
        noisy_targets = simulate_k_targets(train_dataset.targets, simulated_k)
        train_dataset = ProbeDataset(train_dataset.hidden_states, noisy_targets)
        logger.info(f"  Applied simulated k={simulated_k} to training targets")
    
    num_samples = len(train_dataset)
    
    logger.info(f"  Train: {len(train_dataset)} examples, Val: {len(val_dataset)} examples")
    
    # Create probe
    probe = create_probe(args.probe_type, hidden_dim, args)
    
    # Apply standardization if requested
    if args.preprocess:
        mean = train_dataset.hidden_states.mean(dim=0)
        std = train_dataset.hidden_states.std(dim=0) + 1e-8
        probe.set_normalization(mean, std)
        logger.info(f"  Applied standardization (mean range: [{mean.min():.2f}, {mean.max():.2f}])")
    
    # Create training config
    config = TrainingConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        device=args.device,
        loss_type=args.loss,
        optimizer_type=args.optimizer,
        lbfgs_max_iter=args.lbfgs_max_iter,
        lbfgs_history_size=args.lbfgs_history_size,
        apply_sigmoid=args.apply_sigmoid,
        lr_scheduler_type=args.lr_scheduler,
        num_classes=args.num_classes,
    )
    
    # Initialize wandb for this layer
    if use_wandb:
        run_name = f"layer_{layer_idx:02d}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "layer_idx": layer_idx,
                "probe_type": args.probe_type,
                "loss": args.loss,
                "optimizer": args.optimizer,
                "lr": args.lr,
                "lr_scheduler": args.lr_scheduler,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "patience": args.patience,
                "hidden_dim": hidden_dim,
                "preprocess": args.preprocess,
                "num_samples": num_samples,
                "seed": args.seed,
                "apply_sigmoid": args.apply_sigmoid,
                "simulated_k": simulated_k,
                "num_classes": args.num_classes,
                "train_dir": str(train_dir),
                "val_dir": str(val_dir) if val_dir is not None else None,
                "val_num_examples": args.val_num_examples,
            },
            reinit=True,
        )
    
    # Create trainer
    trainer = ProbeTrainer(
        probe=probe,
        config=config,
        output_dir=layer_output_dir,
        use_wandb=use_wandb,
    )
    
    # Train
    results = trainer.train(train_dataset, val_dataset)
    
    # Evaluate on validation set
    eval_metrics = trainer.evaluate(val_dataset)
    
    logger.info(
        f"  Layer {layer_idx} - Val Loss: {results['metrics']['best_val_loss']:.4f}, "
        f"MAE: {eval_metrics['mae']:.4f}, Pearson r: {eval_metrics['pearson_r']:.4f}"
    )
    
    # Log final metrics to wandb
    if use_wandb:
        wandb.log({
            "final/val_loss": results["metrics"]["best_val_loss"],
            "final/best_epoch": results["metrics"]["best_epoch"],
            "final/mae": eval_metrics["mae"],
            "final/mse": eval_metrics["mse"],
            "final/rmse": eval_metrics["rmse"],
            "final/pearson_r": eval_metrics["pearson_r"],
            "final/spearman_r": eval_metrics["spearman_r"],
            "final/ece": eval_metrics["ece"],
        })
        wandb.finish()
    
    # Save final probe
    probe.save(layer_output_dir / "final_probe.pt")
    
    # Save layer results
    layer_results = {
        "layer_idx": layer_idx,
        "training": results["metrics"],
        "evaluation": eval_metrics,
    }
    with open(layer_output_dir / "results.json", "w") as f:
        json.dump(layer_results, f, indent=4)
    
    return layer_results


def train_pooled(
    pooling: str,
    train_dir: Path,
    val_dir: Path | None,
    output_dir: Path,
    args: argparse.Namespace,
    hidden_dim: int,
    use_wandb: bool,
    subsample_indices: torch.Tensor | None = None,
    simulated_k: int | None = None,
    train_split_indices: torch.Tensor | None = None,
    val_split_indices: torch.Tensor | None = None,
) -> dict:
    """Train a probe on hidden states pooled across all layers."""
    logger.info(f"Training probe with {pooling} pooling across layers")
    
    # Create pooled output directory
    pooled_output_dir = output_dir / f"layer_{pooling}_pooled"
    pooled_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets with pooling
    train_dataset = ProbeDataset.from_safetensors_pooled(train_dir, pooling)
    if val_split_indices is not None:
        # Carve validation out of the training set using --val_num_examples
        val_dataset = ProbeDataset(
            train_dataset.hidden_states[val_split_indices],
            train_dataset.targets[val_split_indices],
        )
        train_dataset = ProbeDataset(
            train_dataset.hidden_states[train_split_indices],
            train_dataset.targets[train_split_indices],
        )
    else:
        val_dataset = ProbeDataset.from_safetensors_pooled(val_dir, pooling)
    logger.info(f"  Train dataset: {train_dataset.hidden_states.shape}, {train_dataset.targets.shape}")
    logger.info(f"  Val dataset: {val_dataset.hidden_states.shape}, {val_dataset.targets.shape}")
    
    # Subsample training set if requested (applied after val split)
    if subsample_indices is not None:
        train_dataset = ProbeDataset(
            train_dataset.hidden_states[subsample_indices],
            train_dataset.targets[subsample_indices],
        )
        logger.info(f"  Subsampled train dataset: {train_dataset.hidden_states.shape}, {train_dataset.targets.shape}")
    
    # Apply simulated k to training targets (adds Binomial noise)
    if simulated_k is not None:
        noisy_targets = simulate_k_targets(train_dataset.targets, simulated_k)
        train_dataset = ProbeDataset(train_dataset.hidden_states, noisy_targets)
        logger.info(f"  Applied simulated k={simulated_k} to training targets")
    
    num_samples = len(train_dataset)
    
    logger.info(f"  Train: {len(train_dataset)} examples, Val: {len(val_dataset)} examples")
    
    # Create probe
    probe = create_probe(args.probe_type, hidden_dim, args)
    
    # Apply standardization if requested
    if args.preprocess:
        mean = train_dataset.hidden_states.mean(dim=0)
        std = train_dataset.hidden_states.std(dim=0) + 1e-8
        probe.set_normalization(mean, std)
        logger.info(f"  Applied standardization (mean range: [{mean.min():.2f}, {mean.max():.2f}])")
    
    # Create training config
    config = TrainingConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        device=args.device,
        loss_type=args.loss,
        optimizer_type=args.optimizer,
        lbfgs_max_iter=args.lbfgs_max_iter,
        lbfgs_history_size=args.lbfgs_history_size,
        apply_sigmoid=args.apply_sigmoid,
        lr_scheduler_type=args.lr_scheduler,
        num_classes=args.num_classes,
    )
    
    # Initialize wandb for pooled experiment
    if use_wandb:
        run_name = f"layer_{pooling}_pooled"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "layer_idx": f"{pooling}_pooled",
                "probe_type": args.probe_type,
                "loss": args.loss,
                "optimizer": args.optimizer,
                "lr": args.lr,
                "lr_scheduler": args.lr_scheduler,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "patience": args.patience,
                "hidden_dim": hidden_dim,
                "preprocess": args.preprocess,
                "num_samples": num_samples,
                "seed": args.seed,
                "apply_sigmoid": args.apply_sigmoid,
                "simulated_k": simulated_k,
                "num_classes": args.num_classes,
                "train_dir": str(train_dir),
                "val_dir": str(val_dir) if val_dir is not None else None,
                "val_num_examples": args.val_num_examples,
            },
            reinit=True,
        )
    
    # Create trainer
    trainer = ProbeTrainer(
        probe=probe,
        config=config,
        output_dir=pooled_output_dir,
        use_wandb=use_wandb,
    )
    
    # Train
    results = trainer.train(train_dataset, val_dataset)
    
    # Evaluate on validation set
    eval_metrics = trainer.evaluate(val_dataset)
    
    logger.info(
        f"  Pooled {pooling} - Val Loss: {results['metrics']['best_val_loss']:.4f}, "
        f"MAE: {eval_metrics['mae']:.4f}, Pearson r: {eval_metrics['pearson_r']:.4f}"
    )
    
    # Log final metrics to wandb
    if use_wandb:
        wandb.log({
            "final/val_loss": results["metrics"]["best_val_loss"],
            "final/best_epoch": results["metrics"]["best_epoch"],
            "final/mae": eval_metrics["mae"],
            "final/mse": eval_metrics["mse"],
            "final/rmse": eval_metrics["rmse"],
            "final/pearson_r": eval_metrics["pearson_r"],
            "final/spearman_r": eval_metrics["spearman_r"],
            "final/ece": eval_metrics["ece"],
        })
        wandb.finish()
    
    # Save final probe
    probe.save(pooled_output_dir / "final_probe.pt")
    
    # Save pooled results
    pooled_results = {
        "pooling": pooling,
        "training": results["metrics"],
        "evaluation": eval_metrics,
    }
    with open(pooled_output_dir / "results.json", "w") as f:
        json.dump(pooled_results, f, indent=4)
    
    return pooled_results

def main():
    args = parse_args()
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Validate closed_form constraints
    if args.optimizer == "closed_form":
        if args.probe_type != "linear":
            raise ValueError("closed_form optimizer requires --probe_type linear")
        if args.loss != "mse":
            raise ValueError("closed_form optimizer requires --loss mse")
        if args.apply_sigmoid:
            raise ValueError("closed_form optimizer requires --no_apply_sigmoid")

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir) if args.val_dir is not None else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get dataset metadata
    n_layers = ProbeDataset.get_num_layers(train_dir)
    hidden_dim = ProbeDataset.get_hidden_dim(train_dir)
    
    # Auto-detect num_classes for linear_classifier from ground_truth_summary.json
    if args.loss == "ce" and args.num_classes is None:
        k_samples = load_k_samples_from_summary(train_dir)
        args.num_classes = k_samples + 1
        logger.info(f"Auto-detected num_classes={args.num_classes} (k_samples={k_samples}) from ground_truth_summary.json")
    
    logger.info(f"Training probes for {n_layers} layers (hidden_dim={hidden_dim})")
    logger.info(f"  Probe type: {args.probe_type}")
    logger.info(f"  Loss: {args.loss}")
    logger.info(f"  Optimizer: {args.optimizer}")
    logger.info(f"  LR: {args.lr}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Train dir: {train_dir}")
    if val_dir is not None:
        logger.info(f"  Val dir: {val_dir}")
    else:
        logger.info(f"  Val: sampled from training set ({args.val_num_examples} examples, seed={args.seed})")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Seed: {args.seed}")
    if args.loss == "ce":
        logger.info(f"  Num classes: {args.num_classes}")
    if args.k is not None:
        logger.info(f"  Simulated k: {args.k} (adding Binomial noise to targets)")
    
    use_wandb = not args.no_wandb
    
    # If --val_num_examples is set, carve a validation split out of the training set.
    # Use a dedicated seeded generator so the split is reproducible via args.seed
    # independent of any other RNG state.
    train_split_indices = None
    val_split_indices = None
    if args.val_num_examples is not None:
        total_samples = ProbeDataset.get_num_samples(train_dir)
        max_val = int(0.5 * total_samples)
        if args.val_num_examples > max_val:
            raise ValueError(
                f"--val_num_examples ({args.val_num_examples}) must be <= "
                f"0.5 * total training instances ({total_samples}) = {max_val}"
            )
        split_generator = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(total_samples, generator=split_generator)
        val_split_indices = perm[: args.val_num_examples]
        train_split_indices = perm[args.val_num_examples :]
        available_train = len(train_split_indices)
        logger.info(
            f"  Val split: {args.val_num_examples} examples held out from training "
            f"(train pool: {available_train} / {total_samples})"
        )
    else:
        available_train = ProbeDataset.get_num_samples(train_dir)
    
    # Generate subsample indices once (shared across all layers). Indexes into the
    # training pool AFTER the val split (if any), so always safe to use.
    subsample_indices = None
    if args.num_samples is not None:
        if args.num_samples < available_train:
            subsample_indices = torch.randperm(available_train)[:args.num_samples]
            logger.info(f"  Subsampling {args.num_samples} / {available_train} training samples")
    
    # Save experiment config
    experiment_config = {
        "train_dir": str(train_dir),
        "val_dir": str(val_dir) if val_dir is not None else None,
        "val_num_examples": args.val_num_examples,
        "probe_type": args.probe_type,
        "loss": args.loss,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "lr_scheduler": args.lr_scheduler,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "preprocess": args.preprocess,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "apply_sigmoid": args.apply_sigmoid,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "layer_indices": args.layer_indices,
        "simulated_k": args.k,
        "mlp_hidden_dim": args.mlp_hidden_dim if args.probe_type == "mlp" else None,
        "mlp_num_layers": args.mlp_num_layers if args.probe_type == "mlp" else None,
        "mlp_dropout": args.mlp_dropout if args.probe_type == "mlp" else None,
        "num_classes": args.num_classes if args.probe_type == "linear_classifier" else None,
    }
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(experiment_config, f, indent=2)

    # Train pooled experiments
    pooled_results = []
    for pooling in ["mean"]:  #, "max"]:
        result = train_pooled(
            pooling=pooling,
            train_dir=train_dir,
            val_dir=val_dir,
            output_dir=output_dir,
            args=args,
            hidden_dim=hidden_dim,
            use_wandb=use_wandb,
            subsample_indices=subsample_indices,
            simulated_k=args.k,
            train_split_indices=train_split_indices,
            val_split_indices=val_split_indices,
        )
        pooled_results.append(result)

    # Determine which layers to train
    if args.layer_indices is not None:
        if args.layer_indices == "":
            layer_indices = list()
            logger.info(f"Training no layers")
        else:
            layer_indices = [int(idx.strip()) for idx in args.layer_indices.split(",")]
            logger.info(f"Training only layers: {layer_indices}")
    else:
        layer_indices = list(range(n_layers))
        logger.info(f"Training all layers: {layer_indices}")
    
    # Train probe for each layer
    all_results = []
    for layer_idx in layer_indices:
        layer_results = train_layer(
            layer_idx=layer_idx,
            train_dir=train_dir,
            val_dir=val_dir,
            output_dir=output_dir,
            args=args,
            hidden_dim=hidden_dim,
            use_wandb=use_wandb,
            subsample_indices=subsample_indices,
            simulated_k=args.k,
            train_split_indices=train_split_indices,
            val_split_indices=val_split_indices,
        )
        all_results.append(layer_results)

    # Print summary
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    
    # Find best layer by validation loss (only if we trained any layers)
    if all_results:
        best_layer = min(all_results, key=lambda x: x["training"]["best_val_loss"])
        logger.info(f"Best layer by val loss: {best_layer['layer_idx']} (loss={best_layer['training']['best_val_loss']:.4f})")
        
        # Find best layer by Pearson r
        best_layer_pearson = max(all_results, key=lambda x: x["evaluation"]["pearson_r"])
        logger.info(f"Best layer by Pearson r: {best_layer_pearson['layer_idx']} (r={best_layer_pearson['evaluation']['pearson_r']:.4f})")
    else:
        best_layer = None
        best_layer_pearson = None
    
    # Log pooled results
    for p_result in pooled_results:
        logger.info(
            f"Pooled {p_result['pooling']} - Val Loss: {p_result['training']['best_val_loss']:.4f}, "
            f"Pearson r: {p_result['evaluation']['pearson_r']:.4f}"
        )
    
    # Save summary of all layers, pooled, and best layer info
    summary = {
        "config": experiment_config,
        "layers": all_results,
        "pooled": pooled_results,
        "best_layer_by_val_loss": {
            "layer_idx": best_layer["layer_idx"],
            "training": best_layer["training"],
            "evaluation": best_layer["evaluation"]
        } if best_layer else None,
        "best_layer_by_pearson_r": {
            "layer_idx": best_layer_pearson["layer_idx"],
            "training": best_layer_pearson["training"],
            "evaluation": best_layer_pearson["evaluation"]
        } if best_layer_pearson else None,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    main()
