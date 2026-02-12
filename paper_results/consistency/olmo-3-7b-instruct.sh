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
        --sampled_path "outputs/${dataset}__Olmo-3-7B-Instruct/sampled.jsonl" \
        --ground_truth_path "outputs/${dataset}__Olmo-3-7B-Instruct/ground_truth.jsonl" \
        --output_dir "estimator_results/consistency/${dataset}__Olmo-3-7B-Instruct" \
        --k_values 5 10 20 \
        --seed 42
done
