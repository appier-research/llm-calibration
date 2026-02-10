# Budget Allocation for Question Answering

Evaluate different budget allocation strategies for question answering tasks. Compares uniform allocation (baseline) against confidence-based allocation.

## Problem Scenario

Given N questions with a fixed total budget (e.g., N×B samples):
- **Baseline**: Allocate uniformly (e.g., B samples per question)
- **Greedy**: Allocate based on confidence - questions with lower confidence get more samples

Evaluation Method: **Best-of-k Strategy**: A question is correct if **ANY** of its allocated samples is correct.

## Quick Start

```bash
python3 evaluate_budget_allocation.py --config default_config.yaml
```

## Configuration

```yaml
datasets:
  - gsm8k-test
  - triviaqa-validation

outputs_dir: ../outputs              # Where to find model outputs
output_dir: ./results                # Where to save results
baseline_samples: 5                  # Samples per question for baseline
random_seed: 42                      # For reproducible sampling
```

### Key Parameters

- `baseline_samples`: Number of samples per question for uniform baseline (default: 5)
- `random_seed`: Ensures reproducible random sampling
- `datasets`: List of datasets to evaluate (script finds all models automatically)
- `confidence_sources`: List of estimated confidence that the **Greedy** algoritm will use
-  Linear interploration between uniform allocation and greedy algorithm (not included in the paper):
    - `search_alphas_for`: select confidence resources
    - `test_alphas`: tested alpha
## Input Data Format

Each model folder should contain:
- `ground_truth.jsonl`:
  - `example_id`: Question identifier
  - `expected_accuracy`: Confidence score (0-1)
  - `ground_truth_answer`: Correct answer
  - `num_samples`: numbers of samples per question
  - `num_correct`: numbers of correct samples per question


## Outputs

### 1. Results CSV: `budget_allocation_results.csv`

Contains for each (dataset, model, method):
- `method`: baseline, oracle, greedy, interpolated_X.XX
- `accuracy`: Dataset accuracy (fraction correct)
- `avg_budget`: Average samples per question
- `std_budget`: Standard deviation of budget
- `min_budget`: Minimum samples allocated
- `max_budget`: Maximum samples allocated
- `alpha`: Interpolation parameter (if applicable)

### 2. Method Comparison Plot: `{dataset}_{model}_budget_sweep.pdf`

These plot **report the experiment results of Figure 5 and Figure 10 in the paper.**

Clustered bar plot comparing three main strategies:
- X-axis: Budget B
- Y-axis: Accuracy
- Bars: baseline, greedy (with different confidence sources)

### 3. Log File: `budget_allocation_log.txt`

Detailed execution log with per-model, per-method results.

### 4. (Optional) Alpha Sweep Plot: `{dataset}_alpha_sweep.png`
* If set 
Line plot showing interpolation analysis between :
- X-axis: Alpha (0=greedy, 1=uniform)
- Y-axis: Accuracy
- Lines: One per model
- Shows how accuracy changes from pure greedy to pure uniform

