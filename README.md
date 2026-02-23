# On Calibration of Large Language Models: From Response To Capability
This is the official codebase for the paper ["On Calibration of Large Language Models: From Response To Capability"](https://drive.google.com/file/d/1my8A3HYPHKAVMmBuCWEChYm1ASob19lc/view?usp=sharing) (Google Drive link, arXiv version coming soon).

## Setup
Install the dependencies:
```bash
pipx install uv
uv sync
```

## Construct Capability Calibration Datasets
To measure how well a method performs under the capability calibration framework, one needs to construct the evaluation datasets first.
Currently, we support the following datasets:
* Factual knowledge: [`triviaqa`](https://huggingface.co/datasets/mandarjoshi/trivia_qa), [`simpleqa-verified`](https://huggingface.co/datasets/google/simpleqa-verified)
* Mathematical reasoning: [`gsm8k`](https://huggingface.co/datasets/openai/gsm8k), [`math-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), [`aime25`](https://huggingface.co/datasets/math-ai/aime25)
* General exams: [`mmlu`](https://huggingface.co/datasets/cais/mmlu), [`gpqa`](https://huggingface.co/datasets/Idavidrein/gpqa)

We use repeated sampling to estimate the expected accuracy of each instance, so you have to specify `k` (the number of samples per instance) in the following command:
```bash
uv run python scripts/construct_eval_datasets.py \
    experiment=default \
    experiment.sampling.k=100 \
    experiment.async_.max_concurrent=500 \
    dataset="gsm8k" \
    dataset.split="test" \
    model="openai_compat" \
    model.model_id="openai/gpt-oss-20b" \
    model.base_url="<your_base_url>" \
    hydra.job_logging.root.level="INFO" \
    inference.max_tokens=32700 \
    inference.temperature=1.0 \
    inference.top_p=1.0 \
    hydra.run.dir="outputs/gsm8k-test__gpt-oss-20b"
```

To reproduce the results in the paper, you can run the following command to construct the evaluation / training datasets:
```bash
models=("olmo-3-7b-instruct" "qwen3-8b-non-thinking" "gpt-oss-20b")
for model in "${models[@]}"; do
    bash paper_results/construct_capability_calibration_datasets/${model}.sh
done
```

Code in these scripts would also construct the training datasets for linear probes.

## Metric Differences
We distinguishes response calibration target (single sampled response correctness) from capability calibration target (expected accuracy of model's output distribution). To visualize the comparison, use the scripts in [`scripts/metric_diff/`](scripts/metric_diff/). See that directory’s [README](scripts/metric_diff/README.md) for usage and configuration. The scripts will scan all models in the constructed dataset folder results from the [Construct Capability Calibration Datasets](#construct-capability-calibration-datasets) section.

To reproduce the results in the paper, you can run the following command to plot the visualization results:
```bash
bash paper_results/metric_diff/main.sh
```


## Confidence Estimation Methods for Capability Calibration
Currently, we support the following confidence estimation methods:
* [Verbalized confidence](#verbalized-confidence): Asking the LLM to state the confidence as a percentage (0-100%).
* [P(True)](#ptrue): Asking the LLM whether it can answer the query correctly by instructing it to respond with only “Yes” or “No”, extract the logprobs of these two tokens, and use the softmax probability of “Yes” as the confidence estimate.
* [Response Consistency](#response-consistency): Set confidence as the frequency of the most frequent answer.

### Verbalized Confidence
As an example, you can run the following command to get verbalized confidence of `Qwen/Qwen3-8B` on the `triviaqa` dataset:
```bash
base_url="<your_base_url>"
uv run python scripts/verbalize_confidence.py \
    --model "Qwen/Qwen3-8B" \
    --base_url ${base_url} \
    --temperature 0.7 \
    --top_p 0.8 \
    --max_concurrent 1000 \
    --ground_truth_jsonl "outputs/triviaqa-validation__Qwen3-8B-non-thinking/ground_truth.jsonl" \
    --output_dir "estimator_results/verbalized_confidence/triviaqa-validation__Qwen3-8B-non-thinking" \
    --max_completion_tokens 5000
```

### P(True)
As an example, you can run the following command to get P(True) of `allenai/Olmo-3-7B-Instruct` on the `triviaqa` dataset:
```bash
base_url="<your_base_url>"
uv run python scripts/ptrue.py \
    --model "allenai/Olmo-3-7B-Instruct" \
    --base_url ${base_url} \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_concurrent 1000 \
    --ground_truth_jsonl "outputs/triviaqa-validation__Olmo-3-7B-Instruct/ground_truth.jsonl" \
    --output_dir "estimator_results/ptrue/triviaqa-validation__Olmo-3-7B-Instruct" \
    --max_completion_tokens 1  # should change to ~8192 for reasoning LMs (e.g., gpt-oss-20b)
```
### Response Consistency
As an example, you can run the following command to get response-consistency confidence of `openai/gpt-oss-20b` on the `triviaqa` dataset:
```bash
uv run python scripts/consistency.py \
    --sampled_path "outputs/triviaqa-validation__gpt-oss-20b/sampled.jsonl" \
    --ground_truth_path "outputs/triviaqa-validation__gpt-oss-20b/ground_truth.jsonl" \
    --output_dir "estimator_results/consistency/triviaqa-validation__gpt-oss-20b" \
    --k_values 5 10 20 \
    --seed 42  # random seed
```

### Training Light-Weight (Linear) Probes

## Applications

### Pass@k Prediction
We shows that capability-calibrated confidence can simulate pass@k performance, predicting multi-sampling performance without additional sampling. To run the simulation, use the scripts in [`applications/passk_simulation/`](applications/passk_simulation/). See that directory’s [README](applications/passk_simulation/README.md) for usage and configuration.

We support two types of simulated pass@k evaluation. For the instance level pass@k performance evaluation, the evaluation metric is the Mean Square Error (MSE). You can run the following command to get the evaluated results, and modify the [`config`](applications/passk_simulation/config/instance_passk_evaluation.yaml) to specify the configuration:
```bash
cd applications/passk_simulation

uv run python src/evaluate_instance_passk.py \
    --config config/instance_passk_evaluation.yaml
```

To simulate the dataset-level pass@k curve, you can run the following command to get the simulated curve for math-500 dataset:
```bash
cd applications/passk_simulation

uv run python src/run_simulation.py \
    --config config/passk_simulation_math500.yaml
```
You can further modify the [`config`](applications/passk_simulation/config/passk_simulation_math500.yaml) to specify the configuration.

To reproduce the results in the paper, you can run the following commands:
```bash
bash paper_results/applications/passk_simulation/instance_level.sh
bash paper_results/applications/passk_simulation/dataset_level.sh
```

### Inference Budget Allocation
Capability-calibrated confidence can guide how to allocate a fixed sampling budget across questions (e.g. more samples on low-confidence questions). To run the evaluation, use the scripts in [`applications/budget_allocation/`](applications/budget_allocation/). See that directory’s [README](applications/budget_allocation/README.md) for usage and configuration.

To do the experiment, you can run the following command to do allocation and evaluate the performance across different budget sizes:
```bash
cd applications/budget_allocation

uv run python evaluate_budget_allocation.py \
    --config default_config.yaml
```
You can further modify the [`config`](applications/passk_simulation/config/passk_simulation_math500.yaml) to specify the configuration.

To reproduce the results in the paper, you can run the following command:
```bash
bash paper_results/applications/budget_allocation/main.sh
```