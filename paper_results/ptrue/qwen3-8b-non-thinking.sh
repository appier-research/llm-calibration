base_url="http://TBD/v1"  # TODO: set your own base URL here
datasets=(
    "triviaqa-validation"
    "simpleqa-verified"
    "gsm8k-test"
    "math-500"
    "aime25-test"
    "mmlu-validation"
    "gpqa-diamond"
)
for dataset in "${datasets[@]}"; do
    uv run python scripts/ptrue.py \
        --model "Qwen/Qwen3-8B" \
        --base_url ${base_url} \
        --temperature 0.7 \
        --top_p 0.8 \
        --max_concurrent 1000 \
        --ground_truth_jsonl "outputs/${dataset}__Qwen3-8B-non-thinking/ground_truth.jsonl" \
        --output_dir "estimator_results/ptrue/${dataset}__Qwen3-8B-non-thinking" \
        --max_completion_tokens 1
done