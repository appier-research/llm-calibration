#!/usr/bin/env python3
"""
Compare two evaluation definitions empirically.

Old Definition: Sample one output's correctness (binary 0/1)
New Definition: Use expected accuracy from ground truth (continuous 0-1)

This script shows that these two definitions lead to different conclusions.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
import yaml


# Display name mappings
MODEL_DISPLAY_NAMES = {
    'Olmo-3-7B-Instruct': 'Olmo-3-7B-Instruct',
    'Qwen3-8B-non-thinking': 'Qwen3-8B',
    'gpt-oss-20b': 'gpt-oss-20b'
}

DATASET_DISPLAY_NAMES = {
    'math-500': 'MATH-500',
    'aime25-test': 'AIME25',
    'gsm8k-test': 'GSM8K',
    'gsm8k-train': 'GSM8K',
    'triviaqa-train': 'TriviaQA',
    'triviaqa-validation': 'TriviaQA',
    'mmlu-test': 'MMLU',
    'mmlu-validation': 'MMLU',
    'gpqa-diamond': 'GPQA',
    'gpqa-main': 'GPQA',
    'simpleqa': 'SimpleQA',
    'simpleqa-verified': 'SimpleQA'
}

# Color palette - first three colors from tab10
TAB10 = sns.color_palette("tab10")

MODEL_COLORS = {
    'Olmo-3-7B-Instruct': '#56B4E9',      # First color (blue)
    'Qwen3-8B-non-thinking': "#E69F00",   # Second color (orange)
    'gpt-oss-20b': '#009E73'             # Third color (green)
}


def get_display_name(name: str, mapping: dict) -> str:
    """Get display name from mapping, fallback to original."""
    return mapping.get(name, name)


def get_model_color(model_name: str):
    """Get color for a specific model."""
    return MODEL_COLORS.get(model_name, TAB10[0])  # Default to first color


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to both console and file."""
    log_file = output_dir / "comparison_log.txt"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config or {}


def parse_folder_name(folder_name: str) -> Tuple[str, str]:
    """
    Parse folder name to extract dataset and model.

    Format: {dataset}__{model}__{hardware}
    Example: gsm8k-test__gpt-oss-20b__A40 -> ('gsm8k-test', 'gpt-oss-20b')
    """
    parts = folder_name.split('__')
    if len(parts) >= 2:
        return parts[0], parts[1]
    return folder_name, 'unknown'


def group_folders_by_dataset(outputs_dir: Path, datasets: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Group output folders by dataset.

    Returns:
        Dict mapping dataset -> [(folder_name, model_name), ...]
    """
    grouped = defaultdict(list)

    for folder in outputs_dir.iterdir():
        if not folder.is_dir():
            continue

        # Filter out folders with __no-boxed suffix
        if '__no-boxed' in folder.name:
            continue
        
        if '__k-200' in folder.name:
            continue
        
        # skip the merge triviqa and qsm8k folder
        if '--merged' in folder.name:
            continue

        dataset, model = parse_folder_name(folder.name)

        # Only include if dataset is in the requested list
        if dataset in datasets:
            grouped[dataset].append((folder.name, model))

    return dict(grouped)


def load_model_data(folder_path: Path, random_seed: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load data for one model folder.

    Returns:
        old_def: Sampled correctness (binary 0/1)
        new_def: Expected accuracy (continuous 0-1)
        example_ids: List of example IDs
    """
    ground_truth_path = folder_path / "ground_truth.jsonl"
    sampled_path = folder_path / "sampled.jsonl"

    # Load ground truth (new definition)
    ground_truth = {}
    with open(ground_truth_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            ground_truth[data['example_id']] = data['expected_accuracy']

    # Load sampled data (old definition)
    sampled_by_id = defaultdict(list)
    with open(sampled_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            sampled_by_id[data['example_id']].append(data['correctness'])

    # Sample one correctness per example
    rng = random.Random(random_seed)
    old_def = []
    new_def = []
    example_ids = []

    for example_id, expected_acc in ground_truth.items():
        if example_id not in sampled_by_id:
            continue

        # Randomly sample one correctness value
        correctness = rng.choice(sampled_by_id[example_id])

        old_def.append(correctness)
        new_def.append(expected_acc)
        example_ids.append(example_id)

    return np.array(old_def), np.array(new_def), example_ids


def compute_metrics(old_def: np.ndarray, new_def: np.ndarray) -> Dict:
    """Compute comparison metrics between two definitions."""
    mae = np.mean(np.abs(new_def - old_def))
    spearman_corr, _ = spearmanr(old_def, new_def)
    kendall_corr, _ = kendalltau(old_def, new_def)

    return {
        'mae': mae,
        'spearman_correlation': spearman_corr,
        'kendall_correlation': kendall_corr,
        'n_examples': len(old_def)
    }


def plot_model_comparison(dataset: str, model_name: str, data: Dict, output_path: Path, color: str = 'C0'):
    """
    Create jittered scatter plot for one model-dataset pair.

    Args:
        dataset: Dataset name
        model_name: Model name
        data: Dict with 'old_def', 'new_def', 'metrics'
        output_path: Path to save the figure
        color: Color for the scatter points
    """
    response_calib = data['old_def']  # Old definition: Response Calibration
    capability_calib = data['new_def']  # New definition: Capability Calibration

    # Get display names
    dataset_display = get_display_name(dataset, DATASET_DISPLAY_NAMES)
    model_display = get_display_name(model_name, MODEL_DISPLAY_NAMES)

    # Create single figure
    fig, ax = plt.subplots(figsize=(6, 3))

    # Add jitter to y-axis (response_calib is 0 or 1)
    jitter_amount = 0.08
    jittered_y = response_calib + np.random.RandomState(42).uniform(-jitter_amount, jitter_amount, len(response_calib))

    # Plot jittered scatter
    ax.scatter(capability_calib, jittered_y, alpha=0.7, s=20, edgecolors='none', color=color)

    # Formatting
    ax.set_xlabel('Capability Calibration', fontsize=14)
    ax.set_ylabel('Response Calibration', fontsize=14)
    ax.set_title(f'{dataset_display} - {model_display}', fontsize=16, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.3, 1.3)
    ax.set_yticks([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, format = 'pdf', dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"  Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare two evaluation definitions empirically'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='default_config.yaml',
        help='Path to config file (default: default_config.yaml)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=None,
        help='Override random seed from config'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Get parameters from config (can be overridden by CLI args)
    outputs_dir = Path(config.get('outputs_dir', '../outputs'))
    output_dir = Path(config.get('output_dir', './results'))
    random_seed = args.random_seed or config.get('random_seed', 42)
    datasets = config.get('datasets', [])

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("=" * 80)
    logger.info("Evaluation Definition Comparison")
    logger.info("=" * 80)
    logger.info(f"Outputs directory: {outputs_dir}")
    logger.info(f"Random seed: {random_seed}")
    logger.info(f"Datasets: {datasets}")

    # Group folders by dataset
    grouped_folders = group_folders_by_dataset(outputs_dir, datasets)
    logger.info(f"Found {len(grouped_folders)} datasets")

    # Collect all metrics for CSV output
    all_metrics = []

    # Process each dataset
    for dataset, folders in sorted(grouped_folders.items()):
        logger.info(f"\nProcessing dataset: {dataset}")
        logger.info(f"  Models: {len(folders)}")

        # Load data and plot each model individually
        for folder_name, model_name in sorted(folders):
            folder_path = outputs_dir / folder_name

            try:
                old_def, new_def, example_ids = load_model_data(folder_path, random_seed)
                metrics = compute_metrics(old_def, new_def)

                # Get display names for CSV
                dataset_display = get_display_name(dataset, DATASET_DISPLAY_NAMES)
                model_display = get_display_name(model_name, MODEL_DISPLAY_NAMES)

                # Record metrics with display names
                all_metrics.append({
                    'dataset': dataset_display,
                    'model': model_display,
                    'n_examples': metrics['n_examples'],
                    'mae': metrics['mae'],
                    'spearman_correlation': metrics['spearman_correlation'],
                    'kendall_correlation': metrics['kendall_correlation']
                })

                logger.info(f"    {model_display}: MAE={metrics['mae']:.4f}, "
                           f"Spearman={metrics['spearman_correlation']:.4f}")

                # Create plot for this model-dataset pair
                model_data = {
                    'old_def': old_def,
                    'new_def': new_def,
                    'metrics': metrics
                }
                # Get color based on model name
                color = get_model_color(model_name)
                plot_path = output_dir / f"{dataset}_{model_name}_comparison.pdf"
                plot_model_comparison(dataset, model_name, model_data, plot_path, color)

            except Exception as e:
                logger.error(f"    Error loading {folder_name}: {e}")
                continue

    # Save metrics to CSV
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        csv_path = output_dir / "comparison_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        logger.info(f"\nMetrics saved to: {csv_path}")

        # Print summary table
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info("\n" + metrics_df.to_string(index=False))

    logger.info("\n" + "=" * 80)
    logger.info("Comparison complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
