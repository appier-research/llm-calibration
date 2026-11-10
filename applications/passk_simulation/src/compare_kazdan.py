"""
Compare probe-based pass@k prediction with Kazdan et al. (2025) under a small sampling budget.

For each random split, every question's responses are divided into a test-time pool and a
held-out set. Kazdan et al.'s dynamic sampler allocates a total budget of b * m samples from the
pools, and the same allocation feeds both Kazdan's beta-binomial estimator and the probe + Bayes
estimator. Ground truth is the unbiased pass@k (Chen et al., 2021) on the held-out responses.

Usage:
    python src/compare_kazdan.py --config config/kazdan_comparison.yaml
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from kazdan_estimators import (
    beta_passk,
    dynamic_sampling,
    fit_beta_binomial,
    probe_bayes_passk,
    probe_plugin_passk,
)
from passk_simulator import pass_at_k
from run_simulation import DATASET_DISPLAY_NAMES, MODEL_DISPLAY_NAMES


METHODS = {
    'probe': {'label': 'Probe', 'color': '#ff7f0e', 'linestyle': '--'},
    'kazdan': {'label': 'Kazdan et al.', 'color': '#1f77b4', 'linestyle': '-.'},
    'probe_bayes': {'label': 'Probe + Bayes', 'color': '#2ca02c', 'linestyle': '-'},
}


def setup_logging(results_dir: Path) -> Path:
    log_path = results_dir / f"comparison_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler(sys.stdout)],
    )
    return log_path


def load_probe_predictions(path: Path):
    """Return probe confidences and per-question outcome vectors (m, num_samples)."""
    rows = [json.loads(line) for line in open(path) if line.strip()]
    confidence = np.array([r['confidence'] for r in rows])
    num_correct = np.array([r['num_correct'] for r in rows])
    num_samples = {r['num_samples'] for r in rows}
    if len(num_samples) != 1:
        raise ValueError(f"Expected a fixed num_samples per question in {path}, got {num_samples}")
    outcomes = (np.arange(num_samples.pop()) < num_correct[:, None]).astype(int)
    return confidence, outcomes


def heldout_passk(heldout: np.ndarray, k_values: list) -> np.ndarray:
    """Per-question unbiased pass@k on held-out outcomes, shape (m, len(k_values))."""
    n = heldout.shape[1]
    return np.array([[pass_at_k(n, c, k) for k in k_values] for c in heldout.sum(axis=1)])


def run_split(confidence: np.ndarray, outcomes: np.ndarray, config: dict, rng: np.random.Generator):
    """Predict pass@k for every method and budget on one random pool / held-out split."""
    k = np.array(config['k_values'])
    pool_size = config['pool_size']
    outcomes = rng.permuted(outcomes, axis=1)
    pool, heldout = outcomes[:, :pool_size], outcomes[:, pool_size:]
    actual = heldout_passk(heldout, k).mean(axis=0)
    m = len(confidence)

    rows, allocations = [], []

    def record(method, budget, predicted):
        for k_j, predicted_j, actual_j in zip(k, predicted, actual):
            rows.append({
                'method': method,
                'budget': budget,
                'k': int(k_j),
                'predicted': predicted_j,
                'actual': actual_j,
            })

    record('probe', 0, probe_plugin_passk(confidence[:, None], k).mean(axis=0))

    for budget in config['budgets']:
        successes, attempts = dynamic_sampling(pool, budget * m, rng)
        alpha, beta = fit_beta_binomial(successes, attempts)
        record('kazdan', budget, beta_passk(alpha, beta, k))

        probe_bayes = probe_bayes_passk(
            confidence[:, None], successes[:, None], attempts[:, None], config['prior_strength'], k
        )
        record('probe_bayes', budget, probe_bayes.mean(axis=0))

        allocations.append({
            'budget': budget,
            'alpha': alpha,
            'beta': beta,
            'frac_capped': np.mean(attempts == pool_size),
            'max_attempts': attempts.max(),
            'min_attempts': attempts.min(),
        })

    return pd.DataFrame(rows), pd.DataFrame(allocations)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate over splits: dataset-level MSE and bias."""
    error = raw['predicted'] - raw['actual']
    return (
        raw.assign(squared_error=error ** 2, error=error)
        .groupby(['dataset', 'model', 'method', 'budget', 'k'], sort=False)
        .agg(
            mse=('squared_error', 'mean'),
            bias=('error', 'mean'),
            predicted=('predicted', 'mean'),
            actual=('actual', 'mean'),
        )
        .reset_index()
    )


def write_table(summary: pd.DataFrame, config: dict, path: Path):
    """Markdown tables of dataset-level MSE at the table k values, one per dataset."""
    table_k = config['table_k_values']
    header = '| Model | Method | b | ' + ' | '.join(f'pass@{k} MSE ↓' for k in table_k) + ' |'
    divider = '|' + '---|' * (3 + len(table_k))

    lines = []
    for dataset, by_dataset in summary.groupby('dataset', sort=False):
        lines += [f'## Dataset-level MSE: {DATASET_DISPLAY_NAMES.get(dataset, dataset)}', '', header, divider]
        for (model, method, budget), group in by_dataset.groupby(['model', 'method', 'budget'], sort=False):
            values = group.set_index('k')['mse']
            cells = ' | '.join(f'{values[k]:.2e}' for k in table_k)
            lines.append(
                f'| {MODEL_DISPLAY_NAMES.get(model, model)} | {METHODS[method]["label"]} | {budget} | {cells} |'
            )
        lines.append('')

    path.write_text('\n'.join(lines))


def plot_mse_vs_budget(summary: pd.DataFrame, config: dict, path: Path, title: str):
    table_k = config['table_k_values']
    fig, axes = plt.subplots(1, len(table_k), figsize=(5 * len(table_k), 4))

    for ax, k in zip(axes, table_k):
        at_k = summary[summary['k'] == k]
        for method, style in METHODS.items():
            rows = at_k[at_k['method'] == method]
            line = {'color': style['color'], 'linestyle': style['linestyle'], 'label': style['label']}
            if method == 'probe':
                ax.axhline(rows['mse'].iloc[0], **line)
            else:
                ax.plot(rows['budget'], rows['mse'], marker='o', linewidth=2, **line)
        ax.set(title=f'pass@{k}', xlabel='Budget b (samples per question)', yscale='log')
        ax.set_xticks(config['budgets'])
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Dataset-level MSE')
    axes[0].legend()
    fig.suptitle(title, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_passk_curves(raw: pd.DataFrame, config: dict, path: Path, title: str):
    """Dataset-level pass@k curves per budget, with 95% bands across splits."""
    budgets = config['budgets']
    k_values = config['k_values']
    fig, axes = plt.subplots(1, len(budgets), figsize=(5 * len(budgets), 4), sharey=True)
    actual = raw[raw['method'] == 'probe'].groupby('k')['actual'].mean()

    for ax, budget in zip(axes, budgets):
        for method, style in METHODS.items():
            rows = raw[(raw['method'] == method) & (raw['budget'] == (0 if method == 'probe' else budget))]
            predicted = rows.groupby('k')['predicted']
            band = predicted.quantile([0.025, 0.975]).unstack()
            ax.plot(k_values, predicted.mean(), color=style['color'], linestyle=style['linestyle'],
                    linewidth=2, label=style['label'])
            ax.fill_between(k_values, band[0.025], band[0.975], color=style['color'], alpha=0.2)
        ax.plot(k_values, actual, 'ko-', linewidth=2, markersize=5, label='Actual pass@k')
        ax.set_xscale('log', base=2)
        ax.set_xticks(k_values, labels=[str(k) for k in k_values])
        ax.set(title=f'b = {budget}', xlabel='k')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('pass@k')
    axes[0].legend(loc='lower right')
    fig.suptitle(title, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Compare probe-based pass@k prediction with Kazdan et al. (2025)')
    parser.add_argument('--config', type=str, default='config/kazdan_comparison.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = Path(config['results_dir']).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(results_dir)
    probe_results_dir = Path(config['probe_results_dir']).resolve()

    logging.info(f"Config: {args.config}")
    logging.info(f"Budgets: {config['budgets']}, pool size: {config['pool_size']}, "
                 f"prior strength: {config['prior_strength']}, splits: {config['num_splits']}")

    raw_frames, allocation_frames = [], []
    for experiment in config['experiments']:
        dataset = experiment['dataset']
        for model in config['models']:
            path = (probe_results_dir / f"{model}__{experiment['probe_suffix']}" / dataset
                    / 'confidence_predictions.jsonl')
            confidence, outcomes = load_probe_predictions(path)
            if outcomes.shape[1] - config['pool_size'] < max(config['k_values']):
                raise ValueError(f"Held-out set too small for k={max(config['k_values'])}: {path}")
            logging.info(f"\n{dataset} / {model}: {len(confidence)} questions from {path}")

            rng = np.random.default_rng(config['seed'])
            for split in range(config['num_splits']):
                rows, allocations = run_split(confidence, outcomes, config, rng)
                tags = {'dataset': dataset, 'model': model, 'split': split}
                raw_frames.append(rows.assign(**tags))
                allocation_frames.append(allocations.assign(**tags))

            allocation = pd.concat(allocation_frames[-config['num_splits']:]).groupby('budget')
            for budget, group in allocation:
                logging.info(f"  b={budget}: capped questions {group['frac_capped'].mean():.1%}, "
                             f"max attempts {group['max_attempts'].max()}")

    raw = pd.concat(raw_frames, ignore_index=True)
    summary = summarize(raw)
    raw.to_csv(results_dir / 'raw_predictions.csv', index=False, float_format='%.6f')
    pd.concat(allocation_frames, ignore_index=True).to_csv(
        results_dir / 'allocations.csv', index=False, float_format='%.6f')
    summary.to_csv(results_dir / 'summary.csv', index=False, float_format='%.6e')
    write_table(summary, config, results_dir / 'tables.md')

    for (dataset, model), group in summary.groupby(['dataset', 'model'], sort=False):
        name = f"{dataset}__{model}"
        title = f"{MODEL_DISPLAY_NAMES.get(model, model)} on {DATASET_DISPLAY_NAMES.get(dataset, dataset)}"
        plot_mse_vs_budget(group, config, results_dir / f"{name}_mse_vs_budget.pdf", title)
        model_raw = raw[(raw['dataset'] == dataset) & (raw['model'] == model)]
        plot_passk_curves(model_raw, config, results_dir / f"{name}_passk_curves.pdf", title)

    logging.info(f"\nResults saved to: {results_dir}")
    logging.info(f"Log saved to: {log_path}")


if __name__ == '__main__':
    main()
