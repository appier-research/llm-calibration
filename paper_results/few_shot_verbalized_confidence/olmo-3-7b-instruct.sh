#!/usr/bin/env bash
# Few-shot verbalized confidence: test ground_truth from outputs/<test>__<model>/,
# few-shot demos sampled once per run from outputs/<train_pool>__<model>/ground_truth.jsonl.
#
# Fixed seed per model for fair comparison across runs (change only when intentionally ablating).
# Seed 42 for gpt-oss-20b — use a different integer for other model scripts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_JSON="${SCRIPT_DIR}/test_to_train_pool.json"

base_url="${BASE_URL:-http://213.173.103.203:33994/v1}"
model="allenai/Olmo-3-7B-Instruct"
model_name="Olmo-3-7B-Instruct"
few_shot_k="${FEW_SHOT_K:-3}"
# Fixed per model (paper / reproducibility)
few_shot_seed="${FEW_SHOT_SEED:-42}"

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
  train_pool="$(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); k=sys.argv[2]; assert k in m, f'missing key {k!r} in {sys.argv[1]}'; print(m[k])" "${MAP_JSON}" "${dataset}")"
  uv run python scripts/few_shot_verbalized_confidence.py \
    --model "${model}" \
    --base_url "${base_url}" \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_concurrent 1000 \
    --ground_truth_jsonl "outputs/${dataset}__${model_name}/ground_truth.jsonl" \
    --train_ground_truth_jsonl "outputs/${train_pool}__${model_name}/ground_truth.jsonl" \
    --few_shot_k "${few_shot_k}" \
    --few_shot_seed "${few_shot_seed}" \
    --output_dir "estimator_results/few_shot_verbalized_confidence/${dataset}__${model_name}" \
    --max_completion_tokens 5000
done
