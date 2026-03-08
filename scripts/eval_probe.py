#!/usr/bin/env python
"""
Evaluate a trained probe on a validation probing dataset.

Usage (default - use dataset targets):
    uv run python scripts/eval_probe.py \
        --eval_dir outputs/triviaqa-validation__Qwen3-8B-non-thinking__RTXPro6000/probing_dataset \
        --probe_path outputs/triviaqa-train__Qwen3-8B-non-thinking__RTXPro6000__fullset/probes/linear__loss-bce__optimizer-sgd__lr-1e-4__lr_scheduler-none__weight_decay-1e-4__batch_size-32__epochs-100__preprocess-false__apply_sigmoid-false__num_samples-76523/layer_mean_pooled/best_probe.pt \
        --output_dir outputs/eval_results

Usage (evaluate sample_correctness probe against expected_accuracy targets):
    uv run python scripts/eval_probe.py \
        --eval_dir outputs/triviaqa-validation__Qwen3-8B-non-thinking__RTXPro6000/probing_dataset_sample \
        --probe_path outputs/triviaqa-train__Qwen3-8B-non-thinking__RTXPro6000/probes_sample/layer_mean_pooled/best_probe.pt \
        --output_dir outputs/eval_results_ea_target \
        --eval_target_type expected_accuracy \
        --ground_truth_path outputs/triviaqa-validation__Qwen3-8B-non-thinking__RTXPro6000/ground_truth.jsonl
"""

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import save_file

from src.metrics.custom import c_star_metrics
from src.probes.linear import LinearProbe
from src.probes.linear_classifier import LinearClassifierProbe
from src.probes.mlp import MLPProbe
from src.probes.trainer import ProbeDataset


@dataclass
class ProbeConfidencePrediction:
    example_id: str
    confidence: float
    expected_accuracy: float
    num_samples: int
    num_correct: int

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained probe on a validation probing dataset"
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Directory with the probing dataset to evaluate the probe on (hidden_states.safetensors, targets.safetensors)",
    )
    parser.add_argument(
        "--probe_path",
        type=str,
        required=True,
        help="Path to the trained probe checkpoint (e.g., best_probe.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for inference (default: 256)",
    )
    parser.add_argument(
        "--eval_target_type",
        type=str,
        choices=["dataset", "expected_accuracy"],
        default="dataset",
        help="Target type for evaluation: 'dataset' uses targets from eval dataset, "
             "'expected_accuracy' uses expected_accuracy from ground_truth.jsonl",
    )
    parser.add_argument(
        "--ground_truth_path",
        type=str,
        default=None,
        help="Path to ground_truth.jsonl (required when --eval_target_type=expected_accuracy)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.eval_target_type == "expected_accuracy" and args.ground_truth_path is None:
        parser.error("--ground_truth_path is required when --eval_target_type=expected_accuracy")
    
    return args


def parse_layer_info(probe_path: Path) -> tuple[str, int | None]:
    """
    Parse layer information from probe path directory name.
    
    Args:
        probe_path: Path to probe checkpoint
    
    Returns:
        Tuple of (layer_type, layer_idx):
        - ("pooled", None) for layer_mean_pooled or layer_max_pooled
        - ("single", layer_idx) for layer_XX
    """
    layer_dir = probe_path.parent.name
    
    # Check for pooled layers
    if "mean_pooled" in layer_dir:
        return "mean", None
    elif "max_pooled" in layer_dir:
        return "max", None
    
    # Check for single layer (e.g., layer_23 or layer_05)
    match = re.match(r"layer_(\d+)", layer_dir)
    if match:
        return "single", int(match.group(1))
    
    raise ValueError(f"Could not parse layer info from directory name: {layer_dir}")


def load_experiment_config(probe_path: Path) -> dict:
    """Load experiment config from the probe's experiment directory."""
    config_path = probe_path.parent.parent / "experiment_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def load_example_ids(eval_dir: Path) -> list[str]:
    """Load example IDs from metadata.json in the eval directory."""
    metadata_path = eval_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {eval_dir}")
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    return metadata["example_ids"]


def load_expected_accuracy_lookup(ground_truth_path: str) -> dict[str, float]:
    """Load expected_accuracy keyed by example_id from ground_truth.jsonl."""
    lookup = {}
    with open(ground_truth_path) as f:
        for line in f:
            item = json.loads(line)
            lookup[item["example_id"]] = item["expected_accuracy"]
    return lookup


def get_expected_accuracy_targets(
    sample_ids: list[str],
    ea_lookup: dict[str, float],
) -> np.ndarray:
    """
    Map sample_ids to expected_accuracy values.
    
    For sample_correctness datasets, sample_id format is "{example_id}_{sampled_id}".
    For expected_accuracy datasets, sample_id is just example_id.
    
    Args:
        sample_ids: List of sample IDs (or example IDs)
        ea_lookup: Dict mapping example_id to expected_accuracy
    
    Returns:
        Array of expected_accuracy values
    """
    targets = []
    for sample_id in sample_ids:
        # Try direct lookup first (for expected_accuracy mode datasets)
        if sample_id in ea_lookup:
            targets.append(ea_lookup[sample_id])
        else:
            # Parse example_id from sample_id (for sample_correctness mode)
            # Format: "{example_id}_{sampled_id}" - split on last underscore
            parts = sample_id.rsplit("_", 1)
            if len(parts) == 2 and parts[0] in ea_lookup:
                targets.append(ea_lookup[parts[0]])
            else:
                raise KeyError(
                    f"Could not find expected_accuracy for sample_id '{sample_id}'. "
                    f"Tried direct lookup and parsing as '{{example_id}}_{{sampled_id}}'."
                )
    return np.array(targets)


def load_probe(probe_path: Path, device: torch.device) -> tuple:
    """
    Load probe from checkpoint.
    
    Returns:
        Tuple of (probe, apply_sigmoid, is_classifier)
        - is_classifier: True if probe is a LinearClassifierProbe
    """
    checkpoint = torch.load(probe_path, weights_only=False)
    class_name = checkpoint.get("class_name", "LinearProbe")
    input_dim = checkpoint["input_dim"]
    
    # Get apply_sigmoid from checkpoint config (MLPProbe) or experiment config (LinearProbe)
    apply_sigmoid = True  # default
    is_classifier = False
    
    if "config" in checkpoint and "apply_sigmoid" in checkpoint["config"]:
        # MLPProbe stores this in checkpoint
        apply_sigmoid = checkpoint["config"]["apply_sigmoid"]
    else:
        # LinearProbe: need to get from experiment config
        exp_config = load_experiment_config(probe_path)
        apply_sigmoid = exp_config.get("apply_sigmoid", True)
    
    # Create probe instance
    if class_name == "MLPProbe":
        config = checkpoint.get("config", {})
        probe = MLPProbe(
            input_dim=input_dim,
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.1),
            activation=config.get("activation", "relu"),
            apply_sigmoid=config.get("apply_sigmoid", True),
        )
    elif class_name == "LinearClassifierProbe":
        config = checkpoint.get("config", {})
        probe = LinearClassifierProbe(
            input_dim=input_dim,
            num_classes=config.get("num_classes", 11),
        )
        is_classifier = True
    else:
        # LinearProbe or default
        probe = LinearProbe(
            input_dim=input_dim,
            apply_sigmoid=apply_sigmoid,
        )
    
    # Load with strict=False to handle None-valued buffers (input_mean, input_std)
    # that aren't in fresh probe's state_dict but exist in checkpoint when preprocess=True
    probe.load_state_dict(checkpoint["state_dict"], strict=False)
    
    # Manually restore normalization buffers if they exist in checkpoint
    state_dict = checkpoint["state_dict"]
    if "input_mean" in state_dict:
        probe.input_mean = state_dict["input_mean"]
    if "input_std" in state_dict:
        probe.input_std = state_dict["input_std"]
    probe.to(device)
    probe.eval()
    
    return probe, apply_sigmoid, is_classifier


def run_inference(
    probe: torch.nn.Module,
    dataset: ProbeDataset,
    apply_sigmoid: bool,
    device: torch.device,
    batch_size: int = 256,
    is_classifier: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on dataset.
    
    Args:
        probe: The probe model
        dataset: Dataset to run inference on
        apply_sigmoid: Whether the probe applies sigmoid (for BCE probes)
        device: Device for inference
        batch_size: Batch size for inference
        is_classifier: Whether the probe is a classifier (uses predict_confidence)
    
    Returns:
        Tuple of (predictions, targets) as numpy arrays
    """
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for hidden_states, targets in loader:
            hidden_states = hidden_states.to(device)
            
            if is_classifier:
                # For classifier probes, use predict_confidence to get scalar [0,1] output
                predictions = probe.predict_confidence(hidden_states)
            else:
                predictions = probe(hidden_states)
                # Apply sigmoid if probe outputs raw logits
                if not apply_sigmoid:
                    predictions = torch.sigmoid(predictions)
            
            all_predictions.extend(predictions.cpu().tolist())
            all_targets.extend(targets.tolist())
    
    return np.array(all_predictions), np.array(all_targets)


def plot_reliability_diagram(
    predictions: np.ndarray,
    targets: np.ndarray,
    ece: float,
    output_path: Path,
    n_bins: int = 10,
) -> None:
    """
    Create and save a reliability diagram for c(x) vs c*(x).
    
    Args:
        predictions: Predicted confidence scores
        targets: Ground truth c*(x) values
        ece: ECE value to display
        output_path: Path to save the figure
        n_bins: Number of bins
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    bin_centers = []
    bin_means = []
    bin_counts = []
    
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (predictions >= bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
        else:
            mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
        
        count = mask.sum()
        bin_counts.append(count)
        
        if count > 0:
            bin_centers.append(predictions[mask].mean())
            bin_means.append(targets[mask].mean())
        else:
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_means.append(np.nan)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect calibration")
    
    # Plot reliability diagram as bar chart
    bar_width = 0.08
    valid_mask = ~np.isnan(bin_means)
    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    
    # Draw bars for each bin
    for i in range(n_bins):
        if bin_counts[i] > 0:
            center = bin_centers[i]
            mean_target = bin_means[i]
            gap = mean_target - center
            
            # Bar from predicted confidence to actual accuracy
            color = "#5778a4" if gap >= 0 else "#e49444"
            ax.bar(
                center,
                mean_target,
                width=bar_width,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.8,
            )
    
    # Plot scatter points for bin centers vs mean targets
    ax.scatter(
        bin_centers[valid_mask],
        bin_means[valid_mask],
        color="red",
        s=50,
        zorder=5,
        marker="o",
        edgecolor="black",
        linewidth=1,
        label="Bin mean c*(x)",
    )
    
    # Add ECE annotation
    ax.text(
        0.05,
        0.95,
        f"ECE = {ece:.4f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    
    # Add count annotation for each bin
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ax.annotate(
                f"n={bin_counts[i]}",
                xy=(bin_centers[i], bin_means[i]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                alpha=0.7,
            )
    
    ax.set_xlabel("Predicted Confidence c(x)", fontsize=12)
    ax.set_ylabel("Mean Ground Truth c*(x)", fontsize=12)
    ax.set_title("Reliability Diagram", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved reliability diagram to {output_path}")


def main():
    args = parse_args()
    
    eval_dir = Path(args.eval_dir)
    probe_path = Path(args.probe_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Evaluating probe: {probe_path}")
    logger.info(f"Eval dataset: {eval_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Device: {device}")
    
    # Parse layer info from probe path
    layer_type, layer_idx = parse_layer_info(probe_path)
    logger.info(f"Layer type: {layer_type}, Layer idx: {layer_idx}")
    
    # Load probe
    probe, apply_sigmoid, is_classifier = load_probe(probe_path, device)
    logger.info(f"Loaded probe (apply_sigmoid={apply_sigmoid}, is_classifier={is_classifier})")
    
    # Load example IDs from metadata
    example_ids = load_example_ids(eval_dir)
    logger.info(f"Loaded {len(example_ids)} example IDs")
    
    # Load evaluation dataset
    if layer_type in ["mean", "max"]:
        dataset = ProbeDataset.from_safetensors_pooled(eval_dir, layer_type)
        logger.info(f"Loaded pooled dataset ({layer_type}): {len(dataset)} examples")
    else:
        dataset = ProbeDataset.from_safetensors(eval_dir, layer_idx)
        logger.info(f"Loaded dataset (layer {layer_idx}): {len(dataset)} examples")
    
    # Verify example IDs match dataset size
    if len(example_ids) != len(dataset):
        raise ValueError(
            f"Mismatch: {len(example_ids)} example IDs vs {len(dataset)} dataset examples"
        )
    
    # Run inference
    predictions, dataset_targets = run_inference(
        probe, dataset, apply_sigmoid, device, args.batch_size, is_classifier
    )
    logger.info(f"Inference complete: {len(predictions)} predictions")
    
    # Override targets with expected_accuracy if requested
    if args.eval_target_type == "expected_accuracy":
        ea_lookup = load_expected_accuracy_lookup(args.ground_truth_path)
        targets = get_expected_accuracy_targets(example_ids, ea_lookup)
        logger.info(f"Using expected_accuracy targets from {args.ground_truth_path}")
    else:
        targets = dataset_targets
        logger.info("Using dataset targets")
    
    # Compute metrics
    metrics = c_star_metrics(predictions.tolist(), targets.tolist())
    
    results = {
        "eval_target_type": args.eval_target_type,
        "mse": metrics.mse,
        "ece": metrics.ece,
        "pearson_r": metrics.pearson_r,
        "spearman_r": metrics.spearman_r,
    }
    if args.eval_target_type == "expected_accuracy":
        results["ground_truth_path"] = args.ground_truth_path
    
    # Save metrics JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Save predicted confidences
    predictions_path = output_dir / "predicted_confidence.safetensors"
    save_file({"predicted_confidence": torch.tensor(predictions)}, predictions_path)
    logger.info(f"Saved predictions to {predictions_path}")
    
    # Save confidence predictions JSONL
    jsonl_path = output_dir / "confidence_predictions.jsonl"
    with open(jsonl_path, "w") as f:
        for eid, conf, exp_acc in zip(example_ids, predictions, targets):
            # Recalculate expected_accuracy to eliminate numerical instability
            num_samples = 100
            num_correct = round(float(exp_acc) * num_samples)
            expected_accuracy = num_correct / num_samples
            
            pred = ProbeConfidencePrediction(
                example_id=eid,
                confidence=float(conf),
                expected_accuracy=expected_accuracy,
                num_samples=num_samples,
                num_correct=num_correct,
            )
            f.write(json.dumps(asdict(pred)) + "\n")
    logger.info(f"Saved {len(example_ids)} predictions to {jsonl_path}")
    
    # Generate reliability diagram
    reliability_path = output_dir / "reliability_diagram.png"
    plot_reliability_diagram(predictions, targets, metrics.ece, reliability_path)
    
    # Print results
    logger.info("=" * 60)
    logger.info("Evaluation Results:")
    logger.info(f"  MSE:        {results['mse']:.6f}")
    logger.info(f"  ECE:        {results['ece']:.6f}")
    logger.info(f"  Pearson r:  {results['pearson_r']:.6f}")
    logger.info(f"  Spearman r: {results['spearman_r']:.6f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
