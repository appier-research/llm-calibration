# On Calibration of Large Language Models: From Response To Capability
This is the official codebase for the paper ["On Calibration of Large Language Models: From Response To Capability"](https://drive.google.com/file/d/1my8A3HYPHKAVMmBuCWEChYm1ASob19lc/view?usp=sharing) (Google Drive link, arXiv version coming soon).

## Setup
Install the dependencies:
```bash
pipx install uv
uv sync
```

## Construct the Evaluation Datasets
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
    experiment.async_.max_concurrent=100 \
    dataset="gsm8k" \
    dataset.split="test" \
    model="openai_compat" \
    model.model_id="openai/gpt-oss-20b" \
    model.base_url="http://194.68.245.55:22055/v1" \
    hydra.job_logging.root.level="INFO" \
    inference.max_tokens=32700 \
    inference.temperature=1.0 \
    inference.top_p=1.0 \
    hydra.run.dir="outputs/gsm8k-test__gpt-oss-20b"
```

## Reproduce the Results
