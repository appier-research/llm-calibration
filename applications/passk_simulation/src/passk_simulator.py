"""
Core functions for pass@k simulation using capability-calibrated confidence.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def load_model_data(folder_path: Path, max_samples: int) -> Tuple[Dict, Dict, List[str]]:
    """
    Load ground truth and construct samples data from ground_truth.jsonl.

    Uses num_samples and num_correct fields from ground_truth.jsonl instead of
    loading the full sampled.jsonl file for efficiency.

    Args:
        folder_path: Path to folder containing ground_truth.jsonl
        max_samples: Maximum number of samples to use per instance

    Returns:
        ground_truth: Dict[example_id, expected_accuracy]
        samples: Dict[example_id, List[correctness]] (synthesized from num_samples/num_correct)
        example_ids: List of valid example IDs
    """
    ground_truth_path = folder_path / "ground_truth.jsonl"

    print(f"grount_truth_path: {ground_truth_path}")

    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Missing ground_truth.jsonl in {folder_path}")

    # Load ground truth and construct samples from num_samples/num_correct
    ground_truth = {}
    samples = {}

    with open(ground_truth_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            example_id = data['example_id']

            # Store expected accuracy
            ground_truth[example_id] = data['expected_accuracy']

            # Construct sample list from num_samples and num_correct
            num_samples = data['num_samples']
            num_correct = data['num_correct']

            # Limit to max_samples if needed
            if num_samples > max_samples:
                # Scale down num_correct proportionally
                num_correct = min(num_correct, max_samples)
                num_samples = max_samples

            # Create synthetic sample list: [1, 1, ..., 1, 0, 0, ..., 0]
            # This preserves the pass@k computation which only needs counts
            sample_list = [1] * num_correct + [0] * (num_samples - num_correct)
            samples[example_id] = sample_list

    # All examples from ground_truth are valid
    example_ids = sorted(ground_truth.keys())

    return ground_truth, samples, example_ids


def load_confidence(confidence_config: dict,
                    folder_path: Path,
                    ground_truth: Dict,
                    example_ids: List[str],
                    estimator_results_dir: Path = None,
                    samples: Dict[str, List[int]] = None) -> Dict[str, float]:
    """
    Flexible confidence loader for different confidence sources.

    Args:
        confidence_config: Configuration dict for this confidence source
        folder_path: Path to model folder
        ground_truth: Ground truth data
        example_ids: Valid example IDs
        estimator_results_dir: Path to estimator results directory
        samples: Dict[example_id, List[correctness]] for oracle_response

    Returns:
        confidence: Dict[example_id, confidence_value]
    """
    conf_type = confidence_config['type']

    if conf_type == 'ground_truth':
        # Upper bound: use expected_accuracy from ground truth
        return {eid: ground_truth[eid] for eid in example_ids}

    elif conf_type == 'oracle_response':
        # Oracle response calibration: randomly pick one sample's correctness
        if samples is None:
            raise ValueError("oracle_response requires samples data")

        seed = confidence_config.get('seed', 42)
        rng = np.random.RandomState(seed)

        confidence = {}
        for eid in example_ids:
            sample_list = samples[eid]
            random_idx = rng.randint(0, len(sample_list))
            confidence[eid] = float(sample_list[random_idx])  # 0.0 or 1.0

        return confidence

    elif conf_type == 'learned':
        # Load from learned confidence files in estimator_results
        if estimator_results_dir is None:
            estimator_results_dir = Path('estimator_results')

        method = confidence_config.get('method', 'verbalized')
        folder_name = folder_path.name

        confidence_map = {}

        if method == 'verbalized':
            # Load from verbalized_confidence
            conf_file = estimator_results_dir / 'verbalized_confidence' / folder_name / 'confidence_predictions.jsonl'

            if not conf_file.exists():
                raise FileNotFoundError(f"Confidence file not found: {conf_file}")

            with open(conf_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    confidence_map[data['example_id']] = data['confidence']

        elif method.startswith('consistency'):
            # Parse k value if specified (e.g., 'consistency_k10')
            if '_k' in method:
                k_value = method.split('_k')[1]
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

        elif method == 'p_true':
            # Load from p_true confidence
            conf_file = estimator_results_dir / 'p_true' / folder_name / 'confidence_predictions.jsonl'

            if not conf_file.exists():
                raise FileNotFoundError(f"Confidence file not found: {conf_file}")

            with open(conf_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    confidence_map[data['example_id']] = data['confidence']

        elif method == 'linear_probe' or method.startswith('linear_probe'):
            # Linear probe uses probe trained on the same model being evaluated
            # Path: estimator_results/linear_probe/{model}__{probe_suffix}/{dataset}/confidence_predictions.jsonl
            #
            # Example:
            #   Evaluating folder: "math-500__Qwen3-8B-non-thinking"
            #   Config probe_suffix: "trained-on-gsm8k-and-triviaqa"
            #   Result path: "estimator_results/linear_probe/Qwen3-8B-non-thinking__trained-on-gsm8k-and-triviaqa/math-500/confidence_predictions.jsonl"

            probe_suffix = confidence_config.get('probe_suffix')
            if not probe_suffix:
                raise ValueError("linear_probe method requires 'probe_suffix' in config")

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

        elif method == 'classifier' or method.startswith('classifier'):
            # Classifier probe
            # Path: estimator_results/classifier/{model}__{probe_suffix}/{dataset}/confidence_predictions.jsonl

            probe_suffix = confidence_config.get('probe_suffix')
            if not probe_suffix:
                raise ValueError("classifier method requires 'probe_suffix' in config")

            # Extract dataset and model from folder_name (format: dataset__model__hardware)
            parts = folder_name.split('__')
            dataset = parts[0]
            model = parts[1] if len(parts) > 1 else 'unknown'

            # Construct full probe model ID: {model}__{probe_suffix}
            probe_model_id = f"{model}__{probe_suffix}"

            conf_file = estimator_results_dir / 'linear_classifier' / probe_model_id / dataset / 'confidence_predictions.jsonl'

            if not conf_file.exists():
                raise FileNotFoundError(f"Confidence file not found: {conf_file}")

            with open(conf_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    confidence_map[data['example_id']] = data['confidence']

        else:
            raise ValueError(f"Unknown learned method: {method}")

        # Extract confidences in order, validate all examples are present
        confidence = {}
        for eid in example_ids:
            if eid not in confidence_map:
                raise ValueError(f"Missing confidence for example: {eid}")
            confidence[eid] = confidence_map[eid]

        return confidence

    else:
        raise ValueError(f"Unknown confidence type: {conf_type}")


def apply_clipping(confidence: Dict[str, float], threshold: float) -> Dict[str, float]:
    """
    Apply confidence clipping: set confidence to 0.0 if below threshold.

    Args:
        confidence: Dict[example_id, confidence_value]
        threshold: Minimum confidence threshold

    Returns:
        clipped_confidence: Dict[example_id, clipped_confidence_value]
    """
    return {eid: (conf if conf >= threshold else 0.0) for eid, conf in confidence.items()}


def simulate_passk(confidence: Dict[str, float],
                   k_values: List[int],
                   example_ids: List[str]) -> Dict:
    """
    Simulate pass@k curve using confidence values.

    For each instance i with confidence p_i:
        S_{i,k} = 1 - (1 - p_i)^k

    Dataset statistics:
        μ_k = (1/N) Σ S_{i,k}
        σ_k = sqrt((1/N²) Σ S_{i,k}(1-S_{i,k}))
        95% CI: μ_k ± 1.96σ_k

    Args:
        confidence: Dict[example_id, confidence_value]
        k_values: List of k values to simulate
        example_ids: List of example IDs

    Returns:
        results: Dict with keys 'k_values', 'mean', 'std', 'ci_lower', 'ci_upper'
    """
    N = len(example_ids)
    results = {
        'k_values': [],
        'mean': [],
        'std': [],
        'ci_lower': [],
        'ci_upper': []
    }

    for k in k_values:
        # Compute S_{i,k} = 1 - (1 - p_i)^k for each instance
        S_i_k = np.array([
            1.0 - (1.0 - confidence[eid])**k
            for eid in example_ids
        ])

        # Mean: μ_k = (1/N) Σ S_{i,k}
        mu_k = np.mean(S_i_k)

        # Standard deviation: σ_k = sqrt((1/N²) Σ S_{i,k}(1-S_{i,k}))
        # This is the standard error of the mean for independent Bernoulli trials
        variance = np.sum(S_i_k * (1.0 - S_i_k)) / (N ** 2)
        sigma_k = np.sqrt(variance)

        # 95% Confidence Interval
        ci_lower = mu_k - 1.96 * sigma_k
        ci_upper = mu_k + 1.96 * sigma_k

        results['k_values'].append(k)
        results['mean'].append(mu_k)
        results['std'].append(sigma_k)
        results['ci_lower'].append(ci_lower)
        results['ci_upper'].append(ci_upper)

    return results


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator for pass@k (numerically stable).

    From: "Evaluating Large Language Models Trained on Code" (Chen et al. 2021)
    Figure 3: Numerically stable script for calculating unbiased estimate of pass@k.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k in pass@k

    Returns:
        Unbiased estimate of pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def compute_actual_passk(samples: Dict[str, List[int]],
                         k_values: List[int],
                         example_ids: List[str]) -> Dict:
    """
    Compute actual pass@k from samples using unbiased estimator.

    Args:
        samples: Dict[example_id, List[correctness]]
        k_values: List of k values
        example_ids: List of example IDs

    Returns:
        results: Dict with keys 'k_values', 'mean', 'skipped_k'
    """
    results = {
        'k_values': [],
        'mean': [],
        'skipped_k': []
    }

    for k in k_values:
        # Check if any instance has n < k
        min_n = min(len(samples[eid]) for eid in example_ids)
        if min_n < k:
            results['skipped_k'].append(k)
            continue

        # Compute pass@k for each instance
        passk_per_instance = []
        for eid in example_ids:
            sample_list = samples[eid]
            n = len(sample_list)
            c = sum(sample_list)

            passk = pass_at_k(n, c, k)
            passk_per_instance.append(passk)

        # Average across all instances
        mean_passk = np.mean(passk_per_instance)

        results['k_values'].append(k)
        results['mean'].append(mean_passk)

    return results


def parse_folder_name(folder_name: str) -> Tuple[str, str, str]:
    """
    Parse folder name in format: dataset__model__hardware

    Args:
        folder_name: Folder name string

    Returns:
        (dataset, model, hardware)
    """
    parts = folder_name.split('__')
    if len(parts) < 2:
        raise ValueError(f"Invalid folder name format: {folder_name}")

    dataset = parts[0]
    model = parts[1]
    hardware = parts[2] if len(parts) > 2 else "unknown"

    return dataset, model, hardware


def group_folders_by_dataset(outputs_dir: Path,
                             datasets: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Group output folders by dataset and auto-discover models.

    Args:
        outputs_dir: Path to outputs directory
        datasets: List of dataset names to include

    Returns:
        grouped: Dict[dataset, List[(folder_name, model)]]
    """
    grouped = defaultdict(list)

    for folder in outputs_dir.iterdir():
        if not folder.is_dir():
            continue

        # Skip folders with '__no-boxed' variant
        if '__no-boxed' in folder.name:
            continue

        if '__k-200' in folder.name:
            continue

        try:
            dataset, model, hardware = parse_folder_name(folder.name)

            if dataset in datasets:
                grouped[dataset].append((folder.name, model))
        except ValueError:
            # Skip folders that don't match expected format
            continue

    return dict(grouped)
