"""
Main script for running pass@k simulation experiments.

Usage:
    python run_simulation.py --config ../config/passk_simulation.yaml
"""

import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from passk_simulator import (
    load_model_data,
    load_confidence,
    simulate_passk,
    compute_actual_passk,
    group_folders_by_dataset,
    apply_clipping
)


LINESTYLES = ['-', '--', '-.', ':']

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
    'simpleqa': 'SimpleQA'
}

# Category-based styling (tab10 colors)
TAB10 = sns.color_palette("tab10")
CATEGORY_STYLES = {
    'probe': {'color': "#56B4E9", 'linestyle': '--'},      # blue, solid
    'standard': {'color': "#E69F00", 'linestyle': '-'},   # red, solid
    'baseline': {'color': "#009E73", 'linestyle': '-.'}    # green, solid
}


def get_display_name(name: str, mapping: dict) -> str:
    """Get display name from mapping, fallback to original."""
    return mapping.get(name, name)


def expand_confidence_sources(confidence_sources: dict, clipping_thresholds: List[float]) -> dict:
    """
    Expand confidence sources with clipping variants.

    Note: ground_truth and oracle_response types are not clipped.

    Args:
        confidence_sources: Original confidence sources config
        clipping_thresholds: List of clipping thresholds to apply

    Returns:
        expanded_sources: Dict with variants for each (source, threshold)
    """
    expanded = {}

    for conf_name, conf_config in confidence_sources.items():
        base_label = conf_config.get('label', conf_name)
        conf_type = conf_config.get('type')
        category = conf_config.get('category', 'probe')  # default to probe

        # Get category-based styling
        style = CATEGORY_STYLES.get(category, CATEGORY_STYLES['probe'])
        base_color = style['color']
        base_linestyle = style['linestyle']

        # Skip clipping for ground_truth and oracle_response types
        if conf_type in ['ground_truth', 'oracle_response']:
            expanded[conf_name] = {
                **conf_config,
                'color': base_color,
                'linestyle': base_linestyle,
                'clipping_threshold': 0.0,
                'base_source': conf_name
            }
            continue

        # Regular sources: apply clipping
        for idx, threshold in enumerate(clipping_thresholds):
            # Generate unique key
            variant_key = f"{conf_name}__clip_{threshold}"

            # Generate display label
            if threshold == 0.0:
                variant_label = base_label
            else:
                variant_label = f"{base_label} [≥{threshold}]"

            # Assign linestyle (cycle through available styles if multiple thresholds)
            if len(clipping_thresholds) > 1:
                linestyle = LINESTYLES[idx % len(LINESTYLES)]
            else:
                linestyle = base_linestyle

            # Create variant config
            expanded[variant_key] = {
                **conf_config,
                'label': variant_label,
                'color': base_color,
                'linestyle': linestyle,
                'clipping_threshold': threshold,
                'base_source': conf_name
            }

    return expanded


def setup_logging(results_dir: Path) -> str:
    """
    Setup logging to both console and file.

    Args:
        results_dir: Directory to save log file

    Returns:
        Log file path
    """
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"simulation_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return str(log_path)


def plot_passk_comparison(simulated_results: Dict[str, dict],
                          actual_results: dict,
                          confidence_sources_config: dict,
                          plot_config: dict,
                          output_path: Path,
                          title: str):
    """
    Plot simulated vs actual pass@k curves with adaptive y-axis.

    Args:
        simulated_results: Dict[confidence_name, simulation_results]
        actual_results: Actual pass@k results
        confidence_sources_config: Configuration for confidence sources
        plot_config: Plot configuration
        output_path: Where to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=tuple(plot_config['figsize']))

    # Plot simulated curves with confidence intervals (no markers)
    for conf_name, conf_config in confidence_sources_config.items():
        if conf_name not in simulated_results:
            continue

        sim = simulated_results[conf_name]
        color = conf_config.get('color', 'blue')
        label = conf_config.get('label', conf_name)
        linestyle = conf_config.get('linestyle', '-')

        # Plot mean line (no marker on simulated curves)
        ax.plot(sim['k_values'], sim['mean'],
                color=color, label=label, linewidth=2, linestyle=linestyle)

        # Plot 95% confidence interval band
        if plot_config.get('show_ci', True):
            ax.fill_between(sim['k_values'],
                           sim['ci_lower'],
                           sim['ci_upper'],
                           color=color,
                           alpha=plot_config.get('alpha_ci', 0.2))

    # Plot actual pass@k (with circle marker)
    if actual_results['k_values']:
        ax.plot(actual_results['k_values'], actual_results['mean'],
                'ko-', label='Actual pass@k', linewidth=2, markersize=8)

    # Formatting
    ax.set_xlabel('k (number of samples)', fontsize=18)
    ax.set_ylabel('pass@k', fontsize=18)
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.legend(fontsize=18, loc='lower right')

    if plot_config.get('grid', True):
        ax.grid(True, alpha=plot_config.get('grid_alpha', 0.3))

    ax.set_xlim(left=0)

    # Adaptive y-axis: compute range across all curves with margin
    if plot_config.get('adaptive_ylim', True):
        # Collect all curve values (actual + all simulated)
        all_values = []

        # Add actual pass@k values
        if actual_results['k_values']:
            all_values.extend(actual_results['mean'])

        # Add all simulated curve values
        for conf_name, sim in simulated_results.items():
            all_values.extend(sim['mean'])

        if all_values:
            # Find global min/max across all curves
            global_min = min(all_values)
            global_max = max(all_values)

            # Add margin (absolute value, not percentage)
            margin = plot_config.get('ylim_margin', 0.05)  # Default 0.05 (5% of [0,1] scale)

            y_min = global_min - margin
            y_max = global_max + margin

            # If data is very close to 1.0, use smaller margin above to avoid too much space
            if global_max > 0.95:
                y_max = 1.03

            # Ensure minimum visible range
            min_range = plot_config.get('ylim_min_range', 0.15)
            if (y_max - y_min) < min_range:
                # Expand to minimum range, centered on data
                data_center = (global_min + global_max) / 2
                y_min = data_center - min_range / 2

            # Only prevent going below 0
            if y_min < 0:
                y_min = 0

            ax.set_ylim([y_min, y_max])

            # Ensure y-axis ticks don't go beyond 1.0
            import matplotlib.ticker as ticker
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
            yticks = ax.get_yticks()
            yticks = yticks[yticks <= 1.0]
            ax.set_yticks(yticks)
        else:
            # Fallback: no data
            ax.set_ylim([0, 1])
    else:
        # Default to [0, 1]
        y_tick_list = plot_config.get('y_tick_list', [0.8, 0.9, 1.0])
        ax.set_yticks(y_tick_list)
        #ax.set_ylim([0, 1])

    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(output_path, format = 'pdf', dpi=300, bbox_inches='tight')
    plt.close()


def save_results_to_csv(simulated_results: Dict[str, dict],
                        actual_results: dict,
                        dataset: str,
                        model: str,
                        output_path: Path) -> List[Dict]:
    """
    Save numerical results to CSV and return data for aggregation.

    Args:
        simulated_results: Dict[confidence_name, simulation_results]
        actual_results: Actual pass@k results
        dataset: Dataset name
        model: Model name
        output_path: Where to save the CSV

    Returns:
        List of result dictionaries (for aggregation)
    """
    # Use display names for CSV
    dataset_display = get_display_name(dataset, DATASET_DISPLAY_NAMES)
    model_display = get_display_name(model, MODEL_DISPLAY_NAMES)

    # Prepare data for CSV
    rows = []

    # Add actual pass@k results
    for k, mean_val in zip(actual_results['k_values'], actual_results['mean']):
        rows.append({
            'dataset': dataset_display,
            'model': model_display,
            'k': k,
            'type': 'actual',
            'method': 'actual',
            'mean': mean_val,
            'std': None,
            'ci_lower': None,
            'ci_upper': None
        })

    # Add simulated results for each confidence source
    for conf_name, sim in simulated_results.items():
        for k, mean_val, std_val, ci_l, ci_u in zip(
            sim['k_values'], sim['mean'], sim['std'],
            sim['ci_lower'], sim['ci_upper']
        ):
            rows.append({
                'dataset': dataset_display,
                'model': model_display,
                'k': k,
                'type': 'simulated',
                'method': conf_name,
                'mean': mean_val,
                'std': std_val,
                'ci_lower': ci_l,
                'ci_upper': ci_u
            })

    # Create DataFrame and save individual CSV
    df = pd.DataFrame(rows)
    df = df.sort_values(['type', 'method', 'k'])
    df.to_csv(output_path, index=False, float_format='%.6f')

    return rows


def process_model(folder_path: Path,
                  folder_name: str,
                  model: str,
                  dataset: str,
                  config: dict,
                  results_dir: Path,
                  estimator_results_dir: Path = None) -> List[Dict]:
    """
    Process a single model: simulate and plot pass@k curves.

    Args:
        folder_path: Path to model folder
        folder_name: Name of the folder
        model: Model name
        dataset: Dataset name
        config: Configuration dict
        results_dir: Directory to save results
        estimator_results_dir: Path to estimator results directory (for learned confidence)

    Returns:
        List of result dictionaries for aggregation (or None if failed)
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
        return None

    # Compute actual pass@k
    k_values = config['k_values']
    actual_results = compute_actual_passk(samples, k_values, example_ids)

    if actual_results['skipped_k']:
        logging.warning(f"  Warning: Skipped k values (insufficient samples): {actual_results['skipped_k']}")

    # Expand confidence sources with clipping variants
    clipping_thresholds = config.get('clipping_thresholds', [0.0])
    expanded_sources = expand_confidence_sources(config['confidence_sources'], clipping_thresholds)

    # Compute simulated pass@k for each confidence source variant
    simulated_results = {}

    for variant_key, variant_config in expanded_sources.items():
        try:
            # Load base confidence (without clipping info)
            base_config = {k: v for k, v in variant_config.items()
                          if k not in ['clipping_threshold', 'base_source']}

            # Special handling for oracle_response: average over 5 seeds
            if variant_config.get('type') == 'oracle_response':
                all_seed_results = []
                for seed in range(5):
                    seed_config = {**base_config, 'seed': seed}
                    confidence = load_confidence(
                        seed_config,
                        folder_path,
                        ground_truth,
                        example_ids,
                        estimator_results_dir,
                        samples
                    )
                    seed_result = simulate_passk(confidence, k_values, example_ids)
                    all_seed_results.append(seed_result)

                # Average across seeds
                averaged_result = {
                    'k_values': all_seed_results[0]['k_values'],
                    'mean': np.mean([r['mean'] for r in all_seed_results], axis=0).tolist(),
                    'std': np.mean([r['std'] for r in all_seed_results], axis=0).tolist(),
                    'ci_lower': np.mean([r['ci_lower'] for r in all_seed_results], axis=0).tolist(),
                    'ci_upper': np.mean([r['ci_upper'] for r in all_seed_results], axis=0).tolist()
                }
                simulated_results[variant_key] = averaged_result

            else:
                # Regular confidence sources
                confidence = load_confidence(
                    base_config,
                    folder_path,
                    ground_truth,
                    example_ids,
                    estimator_results_dir,
                    samples
                )

                # Apply clipping
                threshold = variant_config['clipping_threshold']
                confidence = apply_clipping(confidence, threshold)

                # Simulate pass@k
                simulated_results[variant_key] = simulate_passk(
                    confidence,
                    k_values,
                    example_ids
                )

            logging.info(f"  ✓ Simulated with {variant_config['label']}")

        except NotImplementedError as e:
            logging.info(f"  ⊘ Skipped {variant_config['label']}: Not yet implemented")
            continue
        except Exception as e:
            logging.error(f"  ✗ Error with {variant_config['label']}: {e}")
            continue

    if not simulated_results:
        logging.warning(f"  No simulations completed, skipping output")
        return None

    # Create output directory for this model
    model_results_dir = results_dir / f"{dataset}__{model}"
    model_results_dir.mkdir(parents=True, exist_ok=True)

    # Save plot (use display names in title)
    dataset_display = get_display_name(dataset, DATASET_DISPLAY_NAMES)
    model_display = get_display_name(model, MODEL_DISPLAY_NAMES)
    plot_path = model_results_dir / f"{model_display}_{dataset_display}_passk_curve.pdf"
    plot_passk_comparison(
        simulated_results,
        actual_results,
        expanded_sources,
        config['plot_config'],
        plot_path,
        title=f"{dataset_display} - {model_display}"
    )
    logging.info(f"  → Plot saved: {plot_path}")

    # Save CSV and collect data for aggregation
    csv_path = model_results_dir / "passk_results.csv"
    rows = save_results_to_csv(simulated_results, actual_results, dataset, model, csv_path)
    logging.info(f"  → CSV saved: {csv_path}")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Simulate pass@k curves using capability-calibrated confidence"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./config/passk_simulation.yaml',
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

    # Get estimator results directory (default to 'estimator_results' if not specified)
    estimator_results_dir = Path(config.get('estimator_results_dir', 'estimator_results')).resolve()

    # Setup logging
    log_path = setup_logging(results_dir)

    logging.info("=" * 70)
    logging.info("pass@k Simulation")
    logging.info("=" * 70)
    logging.info(f"Config: {config_path}")
    logging.info(f"Datasets: {config['datasets']}")
    logging.info(f"k values: {config['k_values']}")
    logging.info(f"Max samples: {config['max_samples']}")
    logging.info(f"Outputs dir: {outputs_dir}")
    logging.info(f"Results dir: {results_dir}")
    logging.info(f"Log file: {log_path}")

    # Group folders by dataset
    grouped = group_folders_by_dataset(outputs_dir, config['datasets'])

    if not grouped:
        logging.warning("\nNo matching folders found!")
        return

    total_models = sum(len(v) for v in grouped.values())
    logging.info(f"\nFound {total_models} models across {len(grouped)} datasets")

    # Collect all results for aggregated CSV
    all_results = []

    # Process each dataset and model
    for dataset, folder_list in grouped.items():
        logging.info(f"\n{'='*70}")
        logging.info(f"Dataset: {dataset} ({len(folder_list)} models)")
        logging.info(f"{'='*70}")

        for folder_name, model in folder_list:
            folder_path = outputs_dir / folder_name

            rows = process_model(
                folder_path,
                folder_name,
                model,
                dataset,
                config,
                results_dir,
                estimator_results_dir
            )

            # Collect results for aggregation
            if rows is not None:
                all_results.extend(rows)

    # Save aggregated results
    if all_results:
        aggregated_csv_path = results_dir / "all_results.csv"
        df_all = pd.DataFrame(all_results)
        df_all = df_all.sort_values(['dataset', 'model', 'type', 'method', 'k'])
        df_all.to_csv(aggregated_csv_path, index=False, float_format='%.6f')
        logging.info(f"\n{'='*70}")
        logging.info(f"✓ Aggregated results saved: {aggregated_csv_path}")

    logging.info("\n" + "=" * 70)
    logging.info("✓ Simulation complete!")
    logging.info(f"Results saved to: {results_dir}")
    logging.info(f"Log saved to: {log_path}")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
