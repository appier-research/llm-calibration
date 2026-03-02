model="Qwen/Qwen3-8B"
model_name="Qwen3-8B-non-thinking"
prefixes=(
    "triviaqa-validation"
    "simpleqa-verified"
    "gsm8k-test"
    "math-500"
    "aime25-test"
    "mmlu-validation"
    "gpqa-diamond"
    "gsm8k-train"
    "math-train"
    "math-valid"
    "triviaqa-train"
)
for prefix in "${prefixes[@]}"; do
    uv run python scripts/construct_probing_dataset.py \
        --model_id ${model} \
        --ground_truth_path "outputs/${prefix}__${model_name}/ground_truth.jsonl" \
        --sampled_path "outputs/${prefix}__${model_name}/sampled.jsonl" \
        --output_dir "outputs/${prefix}__${model_name}/probing_dataset" \
        --mode "expected_accuracy" \
        --batch_size 32 \
        --cache_dir "/home/brianckwu/.cache/huggingface/hub"
done