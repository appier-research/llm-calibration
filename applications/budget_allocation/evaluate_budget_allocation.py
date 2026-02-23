#!/usr/bin/env python3
"""
Evaluate budget allocation strategies for question answering with multiple random seeds.

Compares uniform allocation vs confidence-based allocation.
Uses pass@k evaluation: a question is correct if ANY sample passes.

This version runs each experiment with multiple random seeds and averages results
to reduce variance from random sampling.
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
import yaml


# Display name mappings for datasets and models
DATASET_DISPLAY_NAMES = {
    'gsm8k-test': 'GSM8K',
    'triviaqa-validation': 'TriviaQA',
    'mmlu-pro': 'MMLU-Pro',
    'math-500': 'MATH-500',
    'aime25-test': 'AIME25',
}

MODEL_DISPLAY_NAMES = {
    'gpt-oss-20b': 'gpt-oss-20B',
    'Olmo-3-7B-Instruct': 'Olmo-3-7B-Instruct',
    'Qwen3-8B-non-thinking': 'Qwen3-8B',
}


def get_display_name(name: str, name_type: str = 'dataset') -> str:
    """
    Get display name for dataset or model.

    Args:
        name: Raw name from folder
        name_type: 'dataset' or 'model'

    Returns:
        Display name (falls back to original if not in mapping)
    """
    if name_type == 'dataset':
        return DATASET_DISPLAY_NAMES.get(name, name)
    elif name_type == 'model':
        return MODEL_DISPLAY_NAMES.get(name, name)
    return name


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to both console and file."""
    log_file = output_dir / "budget_allocation_log.txt"

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
    """Parse folder name to extract dataset and model."""
    parts = folder_name.split('__')
    if len(parts) >= 2:
        return parts[0], parts[1]
    return folder_name, 'unknown'


def group_folders_by_dataset(outputs_dir: Path, datasets: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    """Group output folders by dataset."""
    grouped = defaultdict(list)

    for folder in outputs_dir.iterdir():
        if not folder.is_dir():
            continue

        if '__no-boxed' in folder.name:
            continue

        # Skip k-200 folders (user specified to ignore these)
        if '__k-200' in folder.name:
            continue

        dataset, model = parse_folder_name(folder.name)

        if dataset in datasets:
            grouped[dataset].append((folder.name, model))

    return dict(grouped)


def load_dataset(folder_path: Path) -> Tuple[Dict, Dict]:
    """
    Load ground truth and construct samples data from ground_truth.jsonl.

    Uses num_samples and num_correct fields from ground_truth.jsonl instead of
    loading the full sampled.jsonl file for efficiency.

    Returns:
        ground_truth: Dict mapping example_id -> {'expected_accuracy', 'answer'}
        samples: Dict mapping example_id -> List[correctness values]
    """
    ground_truth_path = folder_path / "ground_truth.jsonl"

    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Missing ground_truth.jsonl in {folder_path}")

    # Load ground truth and construct samples from num_samples/num_correct
    ground_truth = {}
    samples = {}

    with open(ground_truth_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            example_id = data['example_id']

            # Store ground truth info
            ground_truth[example_id] = {
                'expected_accuracy': data['expected_accuracy'],
                'answer': data.get('ground_truth_answer', None)
            }

            # Construct sample list from num_samples and num_correct
            num_samples = data['num_samples']
            num_correct = data['num_correct']

            # Create synthetic sample list: [1, 1, ..., 1, 0, 0, ..., 0]
            # This preserves pass@k computation which only needs counts
            sample_list = [1] * num_correct + [0] * (num_samples - num_correct)
            samples[example_id] = sample_list

    return ground_truth, samples


def get_confidence(ground_truth: Dict, example_ids: List[str], confidence_source: str,
                   folder_path: Path = None, estimator_results_dir: Path = None,
                   linear_probe_configs: Dict = None) -> np.ndarray:
    """
    Extract confidence scores based on source.

    Args:
        ground_truth: Ground truth data
        example_ids: Ordered list of example IDs
        confidence_source: Source of confidence ('oracle', 'verbalized', 'consistency', 'linear_probe_XXX', etc.)
        folder_path: Path to the output folder (for parsing dataset/model)
        estimator_results_dir: Path to estimator results directory
        linear_probe_configs: Dict mapping probe names to probe_suffix (for linear_probe sources)

    Returns:
        confidences: Array of confidence scores
    """
    if confidence_source == 'oracle':
        # Use ground truth expected accuracy
        return np.array([ground_truth[eid]['expected_accuracy'] for eid in example_ids])

    # For external confidence sources, load from file
    if folder_path is None:
        raise ValueError(f"folder_path required for confidence source: {confidence_source}")

    if estimator_results_dir is None:
        estimator_results_dir = Path('estimator_results')

    # Parse folder name to get the subfolder name
    folder_name = folder_path.name

    # Load confidence from appropriate file
    confidence_map = {}

    if confidence_source == 'verbalized':
        # Load from verbalized_confidence
        conf_file = estimator_results_dir / 'verbalized_confidence' / folder_name / 'confidence_predictions.jsonl'
        if not conf_file.exists():
            raise FileNotFoundError(f"Confidence file not found: {conf_file}")

        with open(conf_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                confidence_map[data['example_id']] = data['confidence']

    elif confidence_source.startswith('consistency'):
        # Parse k value if specified (e.g., 'consistency_k10')
        if '_k' in confidence_source:
            k_value = confidence_source.split('_k')[1]
            conf_filename = f'consistency_k{k_value}.jsonl'
        else:
            # Default to k=10
            conf_filename = 'consistency_k10.jsonl'

        conf_file = estimator_results_dir / 'consistency' / folder_name / conf_filename
        if not conf_file.exists():
            raise FileNotFoundError(f"Confidence file not found: {conf_file}")

        with open(conf_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                confidence_map[data['example_id']] = data['confidence']

    elif confidence_source == 'p_true':
        # Load from p_true confidence
        conf_file = estimator_results_dir / 'p_true' / folder_name / 'confidence_predictions.jsonl'
        if not conf_file.exists():
            raise FileNotFoundError(f"Confidence file not found: {conf_file}")

        with open(conf_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                confidence_map[data['example_id']] = data['confidence']

    elif confidence_source.startswith('linear_probe'):
        # Linear probe: estimator_results/linear_probe/{model}__{probe_suffix}/{dataset}/confidence_predictions.jsonl
        # Extract probe name from confidence_source (e.g., 'linear_probe_math' -> 'math')

        if linear_probe_configs is None:
            raise ValueError("linear_probe_configs required for linear_probe confidence sources")

        probe_name = confidence_source.replace('linear_probe_', '')

        if probe_name not in linear_probe_configs:
            raise ValueError(f"Unknown linear probe config: {probe_name}. Available: {list(linear_probe_configs.keys())}")

        probe_suffix = linear_probe_configs[probe_name]

        # Extract dataset and model from folder_name (format: dataset__model__hardware)
        parts = folder_name.split('__')
        dataset = parts[0]
        model = parts[1] if len(parts) > 1 else 'unknown'

        # Construct full probe model ID: {model}__{probe_suffix}
        probe_model_id = f"{model}__{probe_suffix}"

        conf_file = estimator_results_dir / 'linear_probe' / probe_model_id / dataset / 'confidence_predictions.jsonl'

        if not conf_file.exists():
            raise FileNotFoundError(f"Confidence file not found: {conf_file}")

        with open(conf_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                confidence_map[data['example_id']] = data['confidence']

    else:
        raise ValueError(f"Unsupported confidence source: {confidence_source}")

    # Extract confidences in order
    confidences = []
    for eid in example_ids:
        if eid not in confidence_map:
            raise ValueError(f"Missing confidence for example: {eid}")
        confidences.append(confidence_map[eid])

    return np.array(confidences)


def allocate_budget_greedy(confidences: np.ndarray, total_budget: int,
                           min_budget: int = 0, max_budget: int = 100) -> np.ndarray:
    """
    Allocate budget greedily to maximize Expected Pass@k using marginal gain.

    Gain = p * (1-p)^k where k is current allocation.

    Args:
        confidences: Confidence scores (0-1)
        total_budget: Total budget to allocate
        min_budget: Minimum budget per question
        max_budget: Maximum budget per question

    Returns:
        budget_allocation: Integer budget for each question
    """
    n = len(confidences)

    # Clip probabilities to prevent numerical errors
    p = np.clip(confidences, 1e-6, 0.999999)

    # Start with minimum budget
    k_allocations = np.full(n, min_budget, dtype=int)
    current_total = min_budget * n

    if current_total > total_budget:
        raise ValueError(f"Total budget {total_budget} too small for {n} questions with min_budget={min_budget}")

    # Greedy loop: allocate remaining budget one by one
    while current_total < total_budget:
        # Calculate marginal gain for adding +1 sample to each question
        # Set gain to -inf for questions at max_budget
        current_gains = np.where(
            k_allocations < max_budget,
            p * np.power(1 - p, k_allocations),
            -np.inf
        )

        # Check if all questions at max
        if np.all(current_gains == -np.inf):
            break

        # Give sample to question with highest gain
        best_idx = np.argmax(current_gains)
        k_allocations[best_idx] += 1
        current_total += 1

    return k_allocations


def round_to_integers(float_array: np.ndarray, target_sum: int) -> np.ndarray:
    """
    Round float array to integers while preserving exact sum.

    Args:
        float_array: Array of float values
        target_sum: Desired sum of integers

    Returns:
        int_array: Rounded integer array with sum = target_sum
    """
    # Floor first
    int_array = np.floor(float_array).astype(int)
    remainder = target_sum - int_array.sum()

    # Distribute remainder based on fractional parts
    fractional = float_array - int_array
    top_indices = np.argsort(fractional)[-remainder:]
    int_array[top_indices] += 1

    return int_array


def allocate_budget_interpolated(uniform_budget: np.ndarray, greedy_budget: np.ndarray,
                                 alpha: float, total_budget: int, max_budget: int = 100) -> np.ndarray:
    """
    Interpolate between uniform and greedy allocations.

    final = alpha * uniform + (1 - alpha) * greedy

    Args:
        uniform_budget: Uniform allocation
        greedy_budget: Greedy allocation
        alpha: Interpolation weight (0=greedy, 1=uniform)
        total_budget: Total budget to preserve
        max_budget: Maximum budget per question

    Returns:
        interpolated_budget: Rounded integer allocation
    """
    float_budget = alpha * uniform_budget + (1 - alpha) * greedy_budget
    int_budget = round_to_integers(float_budget, total_budget)
    return np.clip(int_budget, 0, max_budget)


def evaluate_passatk(samples: List[int], budget: int, random_seed: int) -> bool:
    """
    Evaluate using pass@k: correct if ANY sample passes.

    Args:
        samples: List of correctness values (0/1)
        budget: Number of samples to use
        random_seed: Random seed for reproducibility

    Returns:
        correct: True if any selected sample is correct
    """
    rng = random.Random(random_seed)

    # If budget >= available samples, use all
    if budget >= len(samples):
        selected = samples
    else:
        selected = rng.sample(samples, budget)

    # Pass@k: correct if ANY sample passes
    return any(s == 1 for s in selected)


def evaluate_method(ground_truth: Dict, samples: Dict, budget_allocation: np.ndarray,
                   example_ids: List[str], random_seed: int) -> float:
    """
    Evaluate a budget allocation method.

    Args:
        ground_truth: Ground truth data
        samples: Sample data
        budget_allocation: Budget for each question
        example_ids: List of example IDs (ordered)
        random_seed: Random seed

    Returns:
        accuracy: Dataset accuracy (fraction correct)
    """
    correct_count = 0
    total_count = len(example_ids)

    for i, example_id in enumerate(example_ids):
        if example_id not in samples:
            continue

        budget = int(budget_allocation[i])
        is_correct = evaluate_passatk(samples[example_id], budget, random_seed)

        if is_correct:
            correct_count += 1

    return correct_count / total_count if total_count > 0 else 0.0


def evaluate_model(folder_path: Path, budget_multiplier: int, confidence_sources: List[str],
                   search_alphas_for: List[str], test_alphas: List[float], random_seed: int,
                   min_budget: int = 0, max_budget: int = 100,
                   estimator_results_dir: Path = None, linear_probe_configs: Dict = None) -> Dict:
    """
    Evaluate budget allocation methods for a single model at a specific budget level.

    Args:
        folder_path: Path to model output folder
        budget_multiplier: Budget multiplier B (total_budget = B * N)
        confidence_sources: List of confidence sources to evaluate
        search_alphas_for: List of confidence sources to perform alpha search on
        test_alphas: Alpha values for interpolation
        random_seed: Random seed
        min_budget: Minimum samples per question
        max_budget: Maximum samples per question
        estimator_results_dir: Path to estimator results directory
        linear_probe_configs: Dict mapping probe names to probe_suffix (for linear_probe sources)

    Returns:
        results: Dict mapping (method, confidence_source, alpha) -> metrics dict
    """
    # Load data
    ground_truth_path = folder_path / "ground_truth.jsonl"
    logging.info(f"    Loading from: {ground_truth_path}")

    ground_truth, samples = load_dataset(folder_path)

    # Get ordered list of example IDs
    example_ids = sorted(ground_truth.keys())
    n_questions = len(example_ids)
    total_budget = n_questions * budget_multiplier

    logging.info(f"    Loaded {n_questions} questions, total budget: {total_budget}")

    # Uniform budget (for baseline and interpolation)
    uniform_budget_value = min(budget_multiplier, max_budget)
    uniform_budget = np.full(n_questions, uniform_budget_value, dtype=float)

    results = {}

    # Method 1: Uniform allocation
    uniform_budget_allocation = np.full(n_questions, uniform_budget_value)
    uniform_accuracy = evaluate_method(ground_truth, samples, uniform_budget_allocation, example_ids, random_seed)
    results[('uniform', None, None)] = {
        'method': 'uniform',
        'confidence_source': None,
        'alpha': None,
        'accuracy': uniform_accuracy,
        'avg_budget': uniform_budget_allocation.mean(),
        'std_budget': uniform_budget_allocation.std(),
        'min_budget': uniform_budget_allocation.min(),
        'max_budget': uniform_budget_allocation.max()
    }

    # Method 2: Greedy for each confidence source
    for conf_source in confidence_sources:
        # Get confidence scores
        confidences = get_confidence(ground_truth, example_ids, conf_source,
                                     folder_path=folder_path,
                                     estimator_results_dir=estimator_results_dir,
                                     linear_probe_configs=linear_probe_configs)

        # Evaluate pure greedy (alpha=0.0)
        greedy_budget = allocate_budget_greedy(confidences, total_budget, min_budget, max_budget)
        greedy_accuracy = evaluate_method(ground_truth, samples, greedy_budget, example_ids, random_seed)
        results[('greedy', conf_source, 0.0)] = {
            'method': 'greedy',
            'confidence_source': conf_source,
            'alpha': 0.0,
            'accuracy': greedy_accuracy,
            'avg_budget': greedy_budget.mean(),
            'std_budget': greedy_budget.std(),
            'min_budget': greedy_budget.min(),
            'max_budget': greedy_budget.max()
        }

        # Alpha search (if requested for this confidence source)
        if conf_source in search_alphas_for:
            for alpha in test_alphas:
                interp_budget = allocate_budget_interpolated(uniform_budget, greedy_budget.astype(float),
                                                             alpha, total_budget, max_budget)
                interp_accuracy = evaluate_method(ground_truth, samples, interp_budget, example_ids, random_seed)
                results[('greedy', conf_source, alpha)] = {
                    'method': 'greedy',
                    'confidence_source': conf_source,
                    'alpha': alpha,
                    'accuracy': interp_accuracy,
                    'avg_budget': interp_budget.mean(),
                    'std_budget': interp_budget.std(),
                    'min_budget': interp_budget.min(),
                    'max_budget': interp_budget.max()
                }

    return results


def format_method_name(method: str, conf_source: str, alpha: float) -> str:
    """Format method name for display."""
    if method == 'uniform':
        return 'Uniform'

    # Map confidence source to display name
    conf_display = conf_source
    if conf_source == 'oracle':
        conf_display = 'Oracle'
    elif conf_source == 'verbalized':
        conf_display = 'Verbalized'
    elif conf_source == 'linear_probe_math':
        conf_display = 'Probe-MATH'
    elif conf_source and conf_source.startswith('linear_probe_'):
        probe_name = conf_source.replace('linear_probe_', '')
        conf_display = f'Probe-{probe_name.upper()}'

    if alpha == 0.0:
        return f'greedy ({conf_display})'
    else:
        return f'greedy ({conf_display}, α={alpha:.2f})'


def plot_budget_sweep(results_df: pd.DataFrame, dataset: str, model: str, output_path: Path, ylim_config: Dict = None):
    """
    Create line plot showing accuracy vs budget multiplier.

    Args:
        results_df: DataFrame with columns [budget_multiplier, method, confidence_source, alpha, accuracy]
        dataset: Dataset name
        model: Model name
        output_path: Path to save plot
        ylim_config: Optional dict with (dataset, model) -> (ymin, ymax) mapping
    """
    if results_df.empty:
        return

    # Create display names
    results_df['display_name'] = results_df.apply(
        lambda row: format_method_name(row['method'], row['confidence_source'], row.get('alpha', 0.0)),
        axis=1
    )

    # Define global style mapping
    styles = {
        "probe":    {"color": "#56B4E9",   "marker": "o", "ls": "-"},
        "standard": {"color": "#E69F00",    "marker": "s", "ls": "-"},
        "baseline": {"color": "#009E73",  "marker": "^", "ls": "-"},
        "oracle":   {"color": "black",      "marker": "v", "ls": "--"}
    }

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique methods and order them: greedy (Oracle) -> greedy (Probe-MATH) -> greedy (Verbalized) -> Uniform
    methods = results_df['display_name'].unique()

    # Define desired order
    desired_order = ['greedy (Oracle)', 'greedy (Probe-MATH)', 'greedy (Verbalized)', 'Uniform']

    # Keep only methods that exist in the data, in the desired order
    sorted_methods = [m for m in desired_order if m in methods]

    # Add any remaining methods not in the desired order (for flexibility)
    remaining = [m for m in methods if m not in sorted_methods]
    sorted_methods.extend(sorted(remaining))

    for i, method_name in enumerate(sorted_methods):
        method_data = results_df[results_df['display_name'] == method_name].sort_values('budget_multiplier')

        # Assign style based on method type
        if 'greedy (Oracle)' in method_name:
            style = styles["oracle"]
        elif 'Probe' in method_name:
            style = styles["probe"]
        elif 'Verbalized' in method_name:
            style = styles["standard"]
        elif method_name == 'Uniform':
            style = styles["baseline"]
        else:
            # Fallback for other methods
            style = styles["standard"]

        # Get marker (handle "None" string)
        marker = style["marker"] if style["marker"] != "None" else None

        ax.plot(method_data['budget_multiplier'], method_data['accuracy'],
                marker=marker, linestyle=style["ls"], linewidth=2,
                color=style["color"], label=method_name, markersize=12)

    # Formatting
    ax.set_xlabel('Compute Budget B', fontsize=24)
    ax.set_ylabel('Success Rate', fontsize=24)
    ax.tick_params(axis='both', labelsize=20)

    # Use display names for title
    dataset_display = get_display_name(dataset, 'dataset')
    model_display = get_display_name(model, 'model')
    ax.set_title(f'{dataset_display} - {model_display}', fontsize=28, fontweight='bold')
    ax.legend(fontsize=24, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    # Reduce y-axis tick density and format to 2 decimal places
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Set y-axis limits
    # Check if custom ylim is specified for this (dataset, model) pair
    if ylim_config and (dataset, model) in ylim_config:
        y_min, y_max = ylim_config[(dataset, model)]
        ax.set_ylim([y_min, y_max])
    else:
        # Adaptive y-axis: compute range across all curves with margin
        # Collect all accuracy values from all methods
        all_values = results_df['accuracy'].values

        if len(all_values) > 0:
            # Find global min/max across all curves
            global_min = all_values.min()
            global_max = all_values.max()

            # Add margin (absolute value)
            margin = 0.05  # 5% of [0,1] scale

            y_min = global_min - margin
            y_max = global_max + margin

            # Ensure minimum visible range
            min_range = 0.15  # 15% of [0,1] scale
            if (y_max - y_min) < min_range:
                # Expand to minimum range, centered on data
                data_center = (global_min + global_max) / 2
                y_min = data_center - min_range / 2
                y_max = data_center + min_range / 2

            # Only prevent going below 0, but allow going above 1.0
            if y_min < 0:
                y_min = 0
                # If we clamped y_min, ensure we still have min_range
                if (y_max - y_min) < min_range:
                    y_max = y_min + min_range

            # Cap display at 1.0 (don't show accuracy > 1.0)
            y_max = min(1.0, y_max)

            ax.set_ylim([y_min, y_max])
        else:
            # Fallback to default
            ax.set_ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"  Saved budget sweep plot: {output_path}")


def plot_alpha_sweep(results_df: pd.DataFrame, dataset: str, confidence_source: str, output_path: Path):
    """
    Create line plot showing accuracy vs alpha for a specific confidence source.

    Args:
        results_df: DataFrame with columns [budget_multiplier, method, confidence_source, alpha, accuracy]
        dataset: Dataset name
        confidence_source: Confidence source to plot (e.g., 'oracle')
        output_path: Path to save plot
    """
    # Filter to this confidence source and methods with alpha
    df_filtered = results_df[
        (results_df['confidence_source'] == confidence_source) &
        (results_df['alpha'].notna())
    ].copy()

    if df_filtered.empty:
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot line for each budget multiplier
    budget_mults = sorted(df_filtered['budget_multiplier'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(budget_mults)))

    for i, budget_mult in enumerate(budget_mults):
        budget_data = df_filtered[df_filtered['budget_multiplier'] == budget_mult].sort_values('alpha')
        ax.plot(budget_data['alpha'], budget_data['accuracy'],
                marker='o', linewidth=2, color=colors[i],
                label=f'B={budget_mult}', markersize=6)

    # Formatting
    ax.set_xlabel('Alpha (0=Pure Greedy, 1=Uniform)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)

    # Use display name for dataset
    dataset_display = get_display_name(dataset, 'dataset')
    ax.set_title(f'Alpha Search: greedy({confidence_source}) - {dataset_display}', fontsize=14, fontweight='bold')
    ax.legend(title='Budget', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, max(df_filtered['alpha']) + 0.05)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"  Saved alpha sweep plot: {output_path}")


def aggregate_seed_results(seed_results: List[Dict]) -> Dict:
    """
    Aggregate results from multiple seeds.

    Args:
        seed_results: List of result dicts from different seeds

    Returns:
        aggregated: Dict with mean and std statistics
    """
    if not seed_results:
        return {}

    # Extract accuracy values
    accuracies = [r['accuracy'] for r in seed_results]

    # Compute statistics
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    # Use first seed for non-statistical fields
    base_result = seed_results[0].copy()

    # Update with aggregated statistics
    base_result['accuracy_mean'] = mean_acc
    base_result['accuracy_std'] = std_acc
    base_result['accuracy'] = mean_acc  # For compatibility with plotting
    base_result['seed_accuracies'] = accuracies

    return base_result


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate budget allocation strategies with multiple random seeds'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='default_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=None,
        help='Override random seed from config'
    )
    parser.add_argument(
        '--num-seeds',
        type=int,
        default=None,
        help='Override number of seeds from config'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Get parameters
    outputs_dir = Path(config.get('outputs_dir', '../outputs'))
    output_dir = Path(config.get('results_dir', './results'))
    estimator_results_dir = Path(config.get('estimator_results_dir', '../estimator_results'))
    random_seed = args.random_seed or config.get('random_seed', 42)
    num_seeds = args.num_seeds or config.get('num_seeds', 5)
    budget_multipliers = config.get('budget_multipliers', [1, 2, 4, 8, 16, 32, 64])
    min_budget = config.get('min_budget', 0)
    max_budget = config.get('max_samples_per_question', 100)
    confidence_sources = config.get('confidence_sources', ['oracle'])
    search_alphas_for = config.get('search_alphas_for', [])
    test_alphas = config.get('test_alphas', [0.25, 0.5, 0.75])
    datasets = config.get('datasets', [])
    linear_probe_configs = config.get('linear_probe_configs', {})
    ylim_config = config.get('ylim_config', {})

    # Convert ylim_config keys from strings to tuples if needed
    # Config format: {"dataset__model": [ymin, ymax]}
    ylim_dict = {}
    for key, value in ylim_config.items():
        if '__' in key:
            dataset, model = key.split('__', 1)
            ylim_dict[(dataset, model)] = tuple(value)
        else:
            logging.warning(f"Invalid ylim_config key format: {key}. Expected 'dataset__model'.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("=" * 80)
    logger.info("Budget Allocation Evaluation (Multi-Seed)")
    logger.info("=" * 80)
    logger.info(f"Outputs directory: {outputs_dir}")
    logger.info(f"Estimator results directory: {estimator_results_dir}")
    logger.info(f"Budget multipliers: {budget_multipliers}")
    logger.info(f"Min budget per question: {min_budget}")
    logger.info(f"Max budget per question: {max_budget}")
    logger.info(f"Confidence sources: {confidence_sources}")
    logger.info(f"Alpha search for: {search_alphas_for}")
    logger.info(f"Test alphas: {test_alphas}")
    logger.info(f"Base random seed: {random_seed}")
    logger.info(f"Number of seeds: {num_seeds}")
    logger.info(f"Seed range: [{random_seed}, {random_seed + num_seeds - 1}]")
    logger.info(f"Datasets: {datasets}")

    # Group folders by dataset
    grouped_folders = group_folders_by_dataset(outputs_dir, datasets)
    logger.info(f"Found {len(grouped_folders)} datasets")

    # Collect all results
    all_results = []

    # Process each dataset
    for dataset, folders in sorted(grouped_folders.items()):
        logger.info(f"\nProcessing dataset: {dataset}")
        logger.info(f"  Models: {len(folders)}")

        # Organize results by model for plotting
        model_results = defaultdict(list)

        # Evaluate each model
        for folder_name, model_name in sorted(folders):
            folder_path = outputs_dir / folder_name
            logger.info(f"  Model: {model_name}")

            # Evaluate at each budget level
            for budget_mult in budget_multipliers:
                logger.info(f"    Budget multiplier: {budget_mult}")

                try:
                    # Run with multiple seeds
                    seed_results_by_key = defaultdict(list)

                    for seed_idx in range(num_seeds):
                        current_seed = random_seed + seed_idx
                        logger.info(f"      Seed {seed_idx + 1}/{num_seeds} (seed={current_seed})")

                        # Evaluate with this seed
                        results = evaluate_model(folder_path, budget_mult, confidence_sources,
                                               search_alphas_for, test_alphas, current_seed,
                                               min_budget, max_budget, estimator_results_dir,
                                               linear_probe_configs)

                        # Collect results by key
                        for key, metrics in results.items():
                            method, conf_source, alpha = key

                            # Format display name
                            display_name = format_method_name(method, conf_source, alpha if alpha is not None else 0.0)

                            result_entry = {
                                'dataset': dataset,
                                'model': model_name,
                                'budget_multiplier': budget_mult,
                                'method': method,
                                'confidence_source': conf_source,
                                'alpha': alpha,
                                'display_name': display_name,
                                'accuracy': metrics['accuracy'],
                                'avg_budget': metrics['avg_budget'],
                                'std_budget': metrics['std_budget'],
                                'min_budget': metrics['min_budget'],
                                'max_budget': metrics['max_budget'],
                                'seed': current_seed
                            }

                            seed_results_by_key[key].append(result_entry)

                            logger.info(f"        {display_name}: {metrics['accuracy']:.4f}")

                    # Aggregate results across seeds
                    logger.info(f"      Aggregating across {num_seeds} seeds:")
                    for key, seed_results in seed_results_by_key.items():
                        method, conf_source, alpha = key

                        # Compute aggregated result
                        aggregated = aggregate_seed_results(seed_results)

                        # Format display name
                        display_name = format_method_name(method, conf_source, alpha if alpha is not None else 0.0)

                        # Log aggregated results
                        mean_acc = aggregated['accuracy_mean']
                        std_acc = aggregated['accuracy_std']
                        logger.info(f"        {display_name}: {mean_acc:.4f} ± {std_acc:.4f}")

                        # Store aggregated result for plotting and CSV
                        all_results.append(aggregated)
                        model_results[model_name].append(aggregated)

                except Exception as e:
                    logger.error(f"    Error evaluating {folder_name} at B={budget_mult}: {e}", exc_info=True)
                    continue

        # Create plots for this dataset
        for model_name, results_list in sorted(model_results.items()):
            plot_df = pd.DataFrame(results_list)

            # Plot 1: Budget sweep (main plot)
            budget_sweep_path = output_dir / f"{dataset}_{model_name}_budget_sweep.pdf"
            plot_budget_sweep(plot_df, dataset, model_name, budget_sweep_path, ylim_dict)

            # Plot 2: Alpha sweep (optional, one per confidence source)
            if search_alphas_for:
                for conf_source in search_alphas_for:
                    alpha_path = output_dir / f"{dataset}_{model_name}_alpha_{conf_source}_sweep.pdf"
                    plot_alpha_sweep(plot_df, dataset, conf_source, alpha_path)

    # Save all results to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        csv_path = output_dir / "budget_allocation_results.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"\nResults saved to: {csv_path}")

        # Print summary (show key columns only)
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)

        # Select key columns for display
        summary_cols = ['dataset', 'model', 'budget_multiplier', 'display_name',
                       'accuracy_mean', 'accuracy_std']
        if all(col in results_df.columns for col in summary_cols):
            summary_df = results_df[summary_cols].copy()
            # Format for better readability
            summary_df['accuracy_mean'] = summary_df['accuracy_mean'].apply(lambda x: f"{x:.4f}")
            summary_df['accuracy_std'] = summary_df['accuracy_std'].apply(lambda x: f"{x:.4f}")
            logger.info("\n" + summary_df.to_string(index=False))
        else:
            logger.info("\n" + results_df.to_string(index=False))

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
