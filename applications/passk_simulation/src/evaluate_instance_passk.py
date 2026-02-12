"""
Evaluate instance-level pass@k predictions.

Usage:
    python evaluate_instance_passk.py --config ../config/instance_passk_evaluation.yaml
"""

import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from passk_simulator import (
    load_model_data,
    load_confidence,
    pass_at_k,
    group_folders_by_dataset
)


def setup_logging(results_dir: Path) -> str:
    """Setup logging to both console and file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"evaluation_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return str(log_path)


def compute_instance_passk_metrics(
    confidence: Dict[str, float],
    samples: Dict[str, List[int]],
    k_values: List[int],
    example_ids: List[str],
    metrics: List[str]
) -> Dict:
    """
    Compute metrics for instance-level pass@k predictions.

    Args:
        confidence: Dict[example_id, confidence_value]
        samples: Dict[example_id, List[correctness]]
        k_values: List of k values
        example_ids: List of example IDs
        metrics: List of metric names to compute

    Returns:
        Dict with structure: {k_value: {metric_name: value}}
    """
    results = {}

    for k in k_values:
        actual_passk = []
        predicted_passk = []

        for eid in example_ids:
            # Actual: use unbiased estimator
            n = len(samples[eid])
            c = sum(samples[eid])
            if n < k:
                continue
            actual = pass_at_k(n, c, k)

            # Predicted: from confidence
            conf = confidence[eid]
            predicted = 1.0 - (1.0 - conf)**k

            actual_passk.append(actual)
            predicted_passk.append(predicted)

        if not actual_passk:
            continue

        actual_arr = np.array(actual_passk)
        predicted_arr = np.array(predicted_passk)

        # Compute requested metrics
        k_metrics = {}
        if "mse" in metrics:
            k_metrics["mse"] = np.mean((actual_arr - predicted_arr)**2)
        if "mae" in metrics:
            k_metrics["mae"] = np.mean(np.abs(actual_arr - predicted_arr))
        if "pearson_r" in metrics:
            k_metrics["pearson_r"] = np.corrcoef(actual_arr, predicted_arr)[0, 1]

        results[k] = k_metrics

    return results


def compute_instance_errors(
    confidence: Dict[str, float],
    samples: Dict[str, List[int]],
    ground_truth: Dict[str, float],
    k: int,
    example_ids: List[str]
) -> tuple:
    """
    Compute prediction errors for each instance at specific k.

    Returns:
        Tuple of (errors, ground_truths) - both arrays aligned by instance
    """
    errors = []
    gts = []

    for eid in example_ids:
        n = len(samples[eid])
        c = sum(samples[eid])
        if n < k:
            continue

        actual = pass_at_k(n, c, k)
        conf = confidence[eid]
        predicted = 1.0 - (1.0 - conf)**k
        errors.append((actual - predicted)**2)  # Squared error
        gts.append(ground_truth[eid])

    return np.array(errors), np.array(gts)


def plot_error_distribution(
    all_errors: Dict[str, tuple],
    k_value: int,
    output_path: Path,
    plot_config: dict
):
    """
    Plot average MSE error as a function of ground truth (expected accuracy).

    Args:
        all_errors: Dict[method_name, (errors_array, ground_truth_array)]
        k_value: K value
        output_path: Where to save plot
        plot_config: Plot configuration
    """
    fig, ax = plt.subplots(figsize=tuple(plot_config['figsize']))

    num_bins = plot_config.get('bins', 50)

    for method_name, (errors, gts) in all_errors.items():
        # Sort by ground truth
        sorted_indices = np.argsort(gts)
        sorted_gts = gts[sorted_indices]
        sorted_errors = errors[sorted_indices]

        # Bin by ground truth and compute average MSE
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_mse = []

        for i in range(num_bins):
            mask = (sorted_gts >= bin_edges[i]) & (sorted_gts < bin_edges[i + 1])
            if i == num_bins - 1:  # Include right edge for last bin
                mask = (sorted_gts >= bin_edges[i]) & (sorted_gts <= bin_edges[i + 1])

            if mask.sum() > 0:
                bin_mse.append(sorted_errors[mask].mean())
            else:
                bin_mse.append(np.nan)

        # Plot line
        ax.plot(bin_centers, bin_mse, marker='o', markersize=3, label=method_name, linewidth=2)

    ax.set_xlabel('Ground Truth (Expected Accuracy)', fontsize=12)
    ax.set_ylabel('Average MSE', fontsize=12)
    ax.set_title(f'Average MSE vs Ground Truth at k={k_value}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=plot_config.get('dpi', 300), bbox_inches='tight')
    plt.close()


def save_results_to_csv(
    all_results: Dict[str, Dict],
    k_values: List[int],
    metrics: List[str],
    model_name: str,
    output_path: Path
):
    """
    Save results to CSV table.

    Args:
        all_results: Dict[method_name, {k: {metric: value}}]
        k_values: List of k values
        metrics: List of metrics
        model_name: Model name
        output_path: Where to save CSV
    """
    rows = []

    for method_name, k_results in all_results.items():
        for metric in metrics:
            row = {'model': model_name, 'method': method_name, 'metric': metric}
            for k in k_values:
                if k in k_results and metric in k_results[k]:
                    row[f'k={k}'] = k_results[k][metric]
                else:
                    row[f'k={k}'] = None
            rows.append(row)

    return rows


def process_model(
    folder_path: Path,
    folder_name: str,
    model: str,
    dataset: str,
    config: dict,
    dataset_results_dir: Path,
    estimator_results_dir: Path = None
):
    """
    Process a single model.

    Returns:
        Tuple of (rows, all_errors) or (None, None) if failed
    """
    logging.info(f"\nProcessing: {folder_name}")

    # Load data
    try:
        ground_truth, samples, example_ids = load_model_data(
            folder_path,
            config['max_samples']
        )
        logging.info(f"  Loaded {len(example_ids)} examples")
    except Exception as e:
        logging.error(f"  Error loading data: {e}")
        return None, None

    # Evaluate each confidence source
    all_results = {}
    all_errors = {}
    k_values = config['k_values']
    metrics = config['metrics']

    for conf_name, conf_config in config['confidence_sources'].items():
        try:
            label = conf_config.get('label', conf_name)

            # Special handling for oracle_response: average over 5 seeds
            if conf_config.get('type') == 'oracle_response':
                all_seed_results = []
                all_seed_errors = []
                all_seed_gts = []

                for seed in range(5):
                    seed_config = {**conf_config, 'seed': seed}
                    confidence = load_confidence(
                        seed_config,
                        folder_path,
                        ground_truth,
                        example_ids,
                        estimator_results_dir,
                        samples
                    )

                    # Compute metrics
                    seed_results = compute_instance_passk_metrics(
                        confidence,
                        samples,
                        k_values,
                        example_ids,
                        metrics
                    )
                    all_seed_results.append(seed_results)

                    # Compute errors
                    errors, gts = compute_instance_errors(confidence, samples, ground_truth, k_values[0], example_ids)
                    all_seed_errors.append(errors)
                    all_seed_gts.append(gts)

                # Average across seeds
                averaged_results = {}
                for k in k_values:
                    averaged_results[k] = {}
                    for metric in metrics:
                        if k in all_seed_results[0] and metric in all_seed_results[0][k]:
                            values = [r[k][metric] for r in all_seed_results if k in r and metric in r[k]]
                            averaged_results[k][metric] = np.mean(values)

                all_results[label] = averaged_results

                # Average errors (use first seed's gts since they're the same)
                averaged_errors = np.mean(all_seed_errors, axis=0)
                all_errors[label] = (averaged_errors, all_seed_gts[0])

            else:
                # Regular confidence sources
                confidence = load_confidence(
                    conf_config,
                    folder_path,
                    ground_truth,
                    example_ids,
                    estimator_results_dir,
                    samples
                )

                # Compute metrics
                results = compute_instance_passk_metrics(
                    confidence,
                    samples,
                    k_values,
                    example_ids,
                    metrics
                )
                all_results[label] = results

                # Compute errors
                errors, gts = compute_instance_errors(confidence, samples, ground_truth, k_values[0], example_ids)
                all_errors[label] = (errors, gts)

            logging.info(f"  ✓ Evaluated {label}")

        except Exception as e:
            logging.error(f"  ✗ Error with {conf_name}: {e}")
            continue

    if not all_results:
        logging.warning(f"  No evaluations completed, skipping output")
        return None, None

    # Generate CSV rows
    rows = save_results_to_csv(all_results, k_values, metrics, model, None)

    # Plot error distribution
    # error_dist_dir = dataset_results_dir / "error_distributions"
    # error_dist_dir.mkdir(parents=True, exist_ok=True)
    # plot_path = error_dist_dir / f"{model}_k{k_values[0]}.png"
    # plot_error_distribution(all_errors, k_values[0], plot_path, config['plot_config'])
    # logging.info(f"  → Plot saved: {plot_path}")

    return rows, all_errors


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate instance-level pass@k predictions"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./config/instance_passk_evaluation.yaml',
        help='Path to configuration YAML file'
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup paths
    outputs_dir = Path(config['outputs_dir']).resolve()
    results_dir = Path(config['results_dir']).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    estimator_results_dir = Path(config.get('estimator_results_dir', 'estimator_results')).resolve()

    # Setup logging
    log_path = setup_logging(results_dir)

    logging.info("=" * 70)
    logging.info("Instance-level pass@k Evaluation")
    logging.info("=" * 70)
    logging.info(f"Config: {config_path}")
    logging.info(f"Datasets: {config['datasets']}")
    logging.info(f"k values: {config['k_values']}")
    logging.info(f"Metrics: {config['metrics']}")
    logging.info(f"Results dir: {results_dir}")
    logging.info(f"Log file: {log_path}")

    # Group folders by dataset
    grouped = group_folders_by_dataset(outputs_dir, config['datasets'])

    if not grouped:
        logging.warning("\nNo matching folders found!")
        return

    total_models = sum(len(v) for v in grouped.values())
    logging.info(f"\nFound {total_models} models across {len(grouped)} datasets")

    # Process each dataset and model
    for dataset, folder_list in grouped.items():
        logging.info(f"\n{'='*70}")
        logging.info(f"Dataset: {dataset} ({len(folder_list)} models)")
        logging.info(f"{'='*70}")

        # Create dataset results directory
        dataset_results_dir = results_dir / dataset
        dataset_results_dir.mkdir(parents=True, exist_ok=True)

        # Collect all rows for this dataset
        all_rows = []

        for folder_name, model in folder_list:
            folder_path = outputs_dir / folder_name

            rows, errors = process_model(
                folder_path,
                folder_name,
                model,
                dataset,
                config,
                dataset_results_dir,
                estimator_results_dir
            )

            if rows is not None:
                all_rows.extend(rows)

        # Save aggregated CSV for this dataset
        if all_rows:
            csv_path = dataset_results_dir / "metrics.csv"
            df = pd.DataFrame(all_rows)
            df.to_csv(csv_path, index=False, float_format='%.4f')
            logging.info(f"\n  → Dataset CSV saved: {csv_path}")

    logging.info("\n" + "=" * 70)
    logging.info("✓ Evaluation complete!")
    logging.info(f"Results saved to: {results_dir}")
    logging.info(f"Log saved to: {log_path}")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
