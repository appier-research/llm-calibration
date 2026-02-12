datasets=(
    "gsm8k-test"
    "triviaqa-validation"
    "mmlu-validation"
    "gpqa-diamond"
    "aime25-test"
    "math-500"
    "simpleqa-verified"
)

for dataset in "${datasets[@]}"; do
    uv run python scripts/consistency.py \
        --sampled_path "outputs/${dataset}__gpt-oss-20b/sampled.jsonl" \
        --ground_truth_path "outputs/${dataset}__gpt-oss-20b/ground_truth.jsonl" \
        --output_dir "estimator_results/consistency/${dataset}__gpt-oss-20b" \
        --k_values 5 10 20 \
        --seed 42
done
