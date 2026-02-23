# pass@k Simulation using Capability-Calibrated Confidence

This project simulates **pass@k curves** using confidence estimates, demonstrating that a good confidence estimator can predict multi-sampling performance without additional computational cost.

## Quick Start

Instance-level pass@k simulation experiment:
```
python src/evaluate_instance_pass@k.py --config ./config/instance_passk_evaluation.yaml
```

Dataset-level pass@k simulation curve on math500:
```
python src/run_simulation.py --config ./config/passk_simulation_math500.yaml
```


## Project Structure

```
passk-simulation/
├── README.md                          # This file
├── config/
|   ├── instance_passk_evaluation.yaml # Instance-level Configuration file
│   └── passk_simulation.yaml          # Dataset-level Configuration file
├── result_analysis.ipynb              # A playground to try different post-hoc calibration methods for dataset-level pass@k simulation curve 
├── src/
|   ├── evaluate_instance_passk.py     # Instance-level simulation analysis
│   ├── passk_simulator.py             # Core simulation functions
│   └── run_simulation.py              # Main execution script
└── results/                           # Output directory 
    ├── {dataset}__{model}/
    │   ├── passk_curve.png            # Per-model visualization
    │   └── passk_results.csv          # Per-model numerical results
    ├── all_results.csv                # Aggregated results from all models
    └── simulation_YYYYMMDD_HHMMSS.log # Execution log
```

## Usage
Here we take dataset-level simulation as an example. Same structure for the instance-level simulation.

### 1. Configure the Experiment

Edit `config/passk_simulation.yaml`:

```yaml
# Specify datasets (script auto-discovers all models)
datasets:
  - gsm8k-test
  - triviaqa-validation

# Paths
outputs_dir: ../../outputs      # Where ground_truth.jsonl and sampled.jsonl are
results_dir: ../results         # Where to save results

# pass@k parameters
k_values: [1, 5, 10, 20, 50, 100]
max_samples: 100

# Confidence sources
confidence_sources:
  upper_bound:
    type: "ground_truth"        # Use expected_accuracy (perfect confidence)
    label: "Upper Bound (Perfect Confidence)"
    color: "#2E86AB"
```

### 2. Run the Simulation

```bash
cd passk-simulation
python src/run_simulation.py --config ./config/passk_simulation_math500.yaml
```


## Input Data Format

Each model folder should contain:
- `ground_truth.jsonl`:
  - `example_id`: Question identifier
  - `expected_accuracy`: Confidence score (0-1)
  - `num_samples`: numbers of samples per question
  - `num_correct`: numbers of correct samples per question


## Output Format 

Log format (`{simulation/evaluation}_YYYYMMDD_HHMMSS.log`)

Complete execution log with:
- Configuration details
- Model processing status
- Warnings and errors
- Output file locations
- Execution summary

### Instance-level simulation results `{dataset}/metrics.csv`

The csv file **reports the experiment results of Table 3 and Table 8 in the paper.**

### Dataset-level simulation Plot Output (`{model}_{dataset}_passk_curve.pdf`)

The plots **report the experiment results of Figure 9 in the paper.**

Visualization showing:
- **Simulated curves**: Mean line with 95% confidence interval bands
- **Actual pass@k**: Black points with lines
- Multiple confidence sources can be overlaid for comparison

## Key Implementation Details

### Unbiased pass@k Estimator

Uses the numerically stable formula from Chen et al. (2021):

```python
def pass_at_k(n, c, k):
    """
    n: total samples
    c: correct samples
    k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

### Handling Insufficient Samples

If any instance has fewer than k samples available, that k value is skipped with a warning.


## Customization

### Change k Values

Edit `config/passk_simulation.yaml`:
```yaml
k_values: [1, 2, 5, 10, 20, 50, 100, 200]
```

### Add More Datasets

```yaml
datasets:
  - gsm8k-test
  - triviaqa-validation
  - mmlu-pro
  - your-custom-dataset
```

The script will automatically discover all models for each dataset.

### Customize Plotting

```yaml
plot_config:
  figsize: [12, 8]
  dpi: 300
  show_ci: true      # Show confidence intervals
  alpha_ci: 0.2      # CI band transparency
  grid: true
  grid_alpha: 0.3
```

## References

- **Chen et al. (2021)**: "Evaluating Large Language Models Trained on Code" - Introduces the unbiased pass@k estimator
