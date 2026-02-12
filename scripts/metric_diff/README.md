# Evaluation Definition Comparison

Empirically demonstrate that two evaluation definitions have different targets when assessing model performance.

## Quick Start

```bash
python3 compare_definitions.py --config default_config.yaml
```

The default config analyzes all available datasets.

## Configuration

Create a YAML config file specifying:

```yaml
# List datasets to analyze (script finds all models automatically)
datasets:
  - gsm8k-test
  - triviaqa-validation
  - mmlu-test

outputs_dir: ../outputs      # Where to find model outputs
output_dir: ./results         # Where to save comparison results
random_seed: 42              # For reproducible sampling
```

## Input Data Format

Each model folder should contain:
- `ground_truth.jsonl`: Contains `example_id` and `expected_accuracy` (Capability Calibration)
- `sampled.jsonl`: Contains `example_id` and `correctness` (multiple samples per example)

The script randomly samples one correctness value per example for Response Calibration.

## Outputs

For each dataset, the script generates:

### 1. Visualization: `{dataset}_{model}_comparison.pdf`

These plots **report the experiment results of Figure 2 and Figure 11 in the paper.**

One figure per model-dataset pair showing a jittered scatter plot:
- X-axis: Capability Calibration (continuous 0-1)
- Y-axis: Response Calibration (binary 0/1, with jitter for visibility)
- Shows how response calibration distributes across capability calibration values
- Title format: "Dataset - Model" (e.g., "gsm8k-test - gpt-oss-20b")
- One figure is generated per model-dataset pair (e.g., `gsm8k-test_gpt-oss-20b_comparison.png`)

### 2. Metrics CSV: `comparison_metrics.csv`

Contains for each model:
- `dataset`, `model`, `n_examples`
- `mae`: Mean Absolute Error between definitions
- `spearman_correlation`: Ranking correlation
- `kendall_correlation`: Alternative ranking correlation

### 3. Log File: `comparison_log.txt`

Complete experiment log with configuration and per-model metrics.

