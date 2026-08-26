#!/usr/bin/env bash
set -euo pipefail

output_root="${OUTPUT_ROOT:-outputs/confidence_training/preliminary_full_weight/seed-42}"
batch_size="${BATCH_SIZE:-16}"
learning_rate="${LEARNING_RATE:-1e-5}"
epochs="${EPOCHS:-3}"
seed="${SEED:-42}"
model_filter="${MODEL_FILTER:-}"
dataset_filter="${DATASET_FILTER:-}"
loss_filter="${LOSS_FILTER:-}"
dry_run="${DRY_RUN:-0}"
wandb_entity="${WANDB_ENTITY:-brian-wu}"
wandb_mode="${WANDB_MODE:-online}"
audit_report="${FULL_WEIGHT_AUDIT_REPORT:-outputs/confidence_training/full_weight_data_audit.json}"

[[ "$batch_size" == "16" ]] || { echo "This protocol fixes BATCH_SIZE=16" >&2; exit 2; }
[[ "$learning_rate" == "1e-5" ]] || { echo "This protocol fixes LEARNING_RATE=1e-5" >&2; exit 2; }
[[ "$epochs" == "3" ]] || { echo "This protocol fixes EPOCHS=3" >&2; exit 2; }
[[ "$seed" == "42" ]] || { echo "This protocol fixes SEED=42" >&2; exit 2; }
[[ "$dry_run" == "0" || "$dry_run" == "1" ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
[[ "$wandb_mode" == "online" || "$wandb_mode" == "offline" || "$wandb_mode" == "disabled" ]] || {
  echo "Invalid WANDB_MODE: $wandb_mode" >&2
  exit 2
}

models=(Olmo-3-7B-Instruct Qwen3-8B-non-thinking)
datasets=(triviaqa math mmlu)
losses=(mse bce ranknet)

if [[ "$dry_run" != "1" ]]; then
  [[ -f "$audit_report" ]] || { echo "Missing adapted-LM audit report: $audit_report" >&2; exit 1; }
  uv run python -c 'import json,sys
value=json.load(open(sys.argv[1]))
if value.get("audit_scope") != "adapted_lm" or value.get("ok") is not True:
    raise SystemExit("Full-weight audit report is absent, wrong-scope, or failed")' "$audit_report"
fi

if [[ "$model_filter" == *gpt-oss* ]]; then
  echo "GPT-OSS is unconditionally excluded from full-weight training" >&2
  exit 2
fi
if [[ -n "$model_filter" && "$model_filter" != "Olmo-3-7B-Instruct" && "$model_filter" != "Qwen3-8B-non-thinking" ]]; then
  echo "Invalid MODEL_FILTER: $model_filter" >&2
  exit 2
fi
if [[ -n "$dataset_filter" && "$dataset_filter" != "triviaqa" && "$dataset_filter" != "math" && "$dataset_filter" != "mmlu" ]]; then
  echo "Invalid DATASET_FILTER: $dataset_filter" >&2
  exit 2
fi
if [[ -n "$loss_filter" && "$loss_filter" != "mse" && "$loss_filter" != "bce" && "$loss_filter" != "ranknet" ]]; then
  echo "Invalid LOSS_FILTER: $loss_filter" >&2
  exit 2
fi

check_input_folder() {
  local folder="$1"
  [[ -f "$folder/ground_truth.jsonl" ]] || { echo "Missing $folder/ground_truth.jsonl" >&2; exit 1; }
  [[ -f "$folder/sampled.jsonl" ]] || { echo "Missing $folder/sampled.jsonl" >&2; exit 1; }
}

check_output_space() {
  local probe="$output_root"
  while [[ ! -e "$probe" && "$probe" != "." && "$probe" != "/" ]]; do
    probe="$(dirname "$probe")"
  done
  [[ -e "$probe" ]] || probe="."
  local available_kib
  available_kib="$(df -Pk "$probe" | awk 'NR==2 {print $4}')"
  local minimum_kib=$((40 * 1024 * 1024))
  (( available_kib >= minimum_kib )) || {
    echo "At least 40 GiB free output space is required before each arm" >&2
    exit 1
  }
}

for model_name in "${models[@]}"; do
  [[ -n "$model_filter" && "$model_name" != "$model_filter" ]] && continue
  case "$model_name" in
    Olmo-3-7B-Instruct)
      model_id="allenai/Olmo-3-7B-Instruct"
      model_revision="6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"
      ;;
    Qwen3-8B-non-thinking)
      model_id="Qwen/Qwen3-8B"
      model_revision="b968826d9c46dd6066d109eabc6255188de91218"
      ;;
  esac

  for dataset_name in "${datasets[@]}"; do
    [[ -n "$dataset_filter" && "$dataset_name" != "$dataset_filter" ]] && continue
    case "$dataset_name" in
      triviaqa)
        train_dir="outputs/triviaqa-train__${model_name}"
        validation_dir="outputs/triviaqa-validation__${model_name}"
        ;;
      math)
        train_dir="outputs/math-train__${model_name}"
        validation_dir="outputs/math-valid__${model_name}"
        ;;
      mmlu)
        train_dir="outputs/mmlu-validation__${model_name}"
        validation_dir="outputs/mmlu-test__${model_name}"
        ;;
    esac
    check_input_folder "$train_dir"
    check_input_folder "$validation_dir"

    for loss in "${losses[@]}"; do
      [[ -n "$loss_filter" && "$loss" != "$loss_filter" ]] && continue
      run_name="${dataset_name}__${model_name}__full_weight__${loss}__lr-1e-5__bs-16__seed-42"
      output_dir="${output_root}/${run_name}"
      manifest="${output_dir}/run_manifest.json"
      if [[ "$dry_run" != "1" && -f "$manifest" ]]; then
        status="$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "$manifest")"
        if [[ "$status" == "complete" ]]; then
          echo "Skipping complete run: ${run_name}" >&2
          continue
        fi
        echo "Refusing existing incomplete run: ${run_name}" >&2
        exit 1
      fi
      if [[ "$dry_run" != "1" && -d "$output_dir" && -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Refusing nonempty run destination: ${output_dir}" >&2
        exit 1
      fi
      check_output_space

      command=(
        uv run python scripts/train_confidence_model.py
        --run-name "$run_name"
        --dataset-name "$dataset_name"
        --model-name "$model_name"
        --model-id "$model_id"
        --model-revision "$model_revision"
        --adaptation-strategy full_weight
        --loss "$loss"
        --train-dir "$train_dir"
        --validation-dir "$validation_dir"
        --output-dir "$output_dir"
        --learning-rate 1e-5
        --batch-size 16
        --epochs 3
        --seed 42
        --device cuda
        --num-workers 0
        --log-every-steps 10
        --gradient-checkpointing
        --no-standardize
        --no-save-resume-state
        --no-resume
        --wandb-project "lm-full-weight__${model_name}__${dataset_name}"
        --wandb-entity "$wandb_entity"
        --wandb-mode "$wandb_mode"
      )
      [[ "$dry_run" == "1" ]] && command+=(--dry-run --print-resolved-config)
      "${command[@]}"
    done
  done
done

if [[ "$dry_run" != "1" && -z "$model_filter" && -z "$dataset_filter" && -z "$loss_filter" ]]; then
  uv run python scripts/summarize_preliminary_confidence_runs.py \
    --root "$output_root" \
    --expected-runs 18 \
    --output-json "$output_root/summary.json" \
    --output-csv "$output_root/summary.csv" \
    --audit-report "$audit_report"
fi
