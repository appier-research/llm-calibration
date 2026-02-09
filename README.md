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
==TODO==

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
As an example, you can run the following command to get verbalized confidence of `openai/gpt-oss-20b` on the `triviaqa` dataset:
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
==TODO==