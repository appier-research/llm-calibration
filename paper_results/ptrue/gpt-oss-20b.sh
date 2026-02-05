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
        --model "openai/gpt-oss-20b" \
        --base_url ${base_url} \
        --temperature 1.0 \
        --top_p 1.0 \
        --max_concurrent 1000 \
        --ground_truth_jsonl "outputs/${dataset}__gpt-oss-20b/ground_truth.jsonl" \
        --output_dir "estimator_results/ptrue/${dataset}__gpt-oss-20b" \
        --max_completion_tokens 5000
done