#!/usr/bin/env bash
set -euo pipefail

# Sequential fixed-config preliminary trajectory for 3 models x 3 datasets x 3 losses.
# Filters: MODEL_FILTER, DATASET_FILTER, LOSS_FILTER. Set DRY_RUN=1 to resolve only.

output_root="${OUTPUT_ROOT:-outputs/confidence_training/preliminary_lora/seed-42}"
epochs="${EPOCHS:-3}"
batch_size="${BATCH_SIZE:-16}"
learning_rate="${LEARNING_RATE:-3e-5}"
log_every_steps="${LOG_EVERY_STEPS:-10}"
seed="${SEED:-42}"
device="${DEVICE:-cuda}"
model_filter="${MODEL_FILTER:-}"
dataset_filter="${DATASET_FILTER:-}"
loss_filter="${LOSS_FILTER:-}"
dry_run="${DRY_RUN:-0}"
wandb_mode="${WANDB_MODE:-online}"
wandb_entity="${WANDB_ENTITY:-}"

if [[ "$batch_size" != "16" ]]; then
  echo "This preliminary protocol fixes BATCH_SIZE=16" >&2
  exit 2
fi

models=(Olmo-3-7B-Instruct Qwen3-8B-non-thinking gpt-oss-20b)
datasets=(triviaqa math mmlu)
losses=(mse bce ranknet)

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
    gpt-oss-20b)
      model_id="openai/gpt-oss-20b"
      model_revision="6cee5e81ee83917806bbde320786a8fb61efebee"
      ;;
  esac

  for dataset_name in "${datasets[@]}"; do
    [[ -n "$dataset_filter" && "$dataset_name" != "$dataset_filter" ]] && continue
    train_template_args=()
    validation_template_args=()
    case "$dataset_name" in
      triviaqa)
        train_dir="outputs/triviaqa-train__${model_name}"
        validation_dir="outputs/triviaqa-validation__${model_name}"
        if [[ "$model_name" == "gpt-oss-20b" ]]; then
          train_template_args=(--train-chat-template-kwargs '{"template_date":"2026-08-23"}')
          validation_template_args=(--validation-chat-template-kwargs '{"template_date":"2026-04-21"}')
        fi
        ;;
      math)
        train_dir="outputs/math-train__${model_name}"
        validation_dir="outputs/math-valid__${model_name}"
        if [[ "$model_name" == "gpt-oss-20b" ]]; then
          train_template_args=(--train-chat-template-kwargs '{"template_date":"2026-04-24"}')
          validation_template_args=(--validation-chat-template-kwargs '{"template_date":"2026-04-24"}')
        fi
        ;;
      mmlu)
        train_dir="outputs/mmlu-validation__${model_name}"
        validation_dir="outputs/mmlu-test__${model_name}"
        if [[ "$model_name" == "gpt-oss-20b" ]]; then
          train_template_args=(--train-chat-template-kwargs '{"template_date":"2026-04-23"}')
          validation_template_args=(--validation-chat-template-kwargs '{"template_date":"2026-04-22"}')
        fi
        ;;
    esac

    for loss in "${losses[@]}"; do
      [[ -n "$loss_filter" && "$loss" != "$loss_filter" ]] && continue
      run_name="${dataset_name}__${model_name}__lora__${loss}__lr-${learning_rate}__bs-${batch_size}__seed-${seed}"
      output_dir="${output_root}/${run_name}"
      manifest="${output_dir}/run_manifest.json"
      if [[ "$dry_run" != "1" && -f "$manifest" ]]; then
        if rg -q '"status": "complete"' "$manifest"; then
          echo "Skipping complete run: ${run_name}"
          continue
        fi
        echo "Refusing existing non-complete run: ${run_name}" >&2
        exit 1
      fi

      command=(
        uv run python scripts/train_confidence_model.py
        --run-name "$run_name"
        --dataset-name "$dataset_name"
        --model-name "$model_name"
        --model-id "$model_id"
        --model-revision "$model_revision"
        --adaptation-strategy lora
        --loss "$loss"
        --train-dir "$train_dir"
        --validation-dir "$validation_dir"
        --output-dir "$output_dir"
        --learning-rate "$learning_rate"
        --batch-size "$batch_size"
        --epochs "$epochs"
        --seed "$seed"
        --device "$device"
        --num-workers 0
        --log-every-steps "$log_every_steps"
        --gradient-checkpointing
        --no-standardize
        --wandb-project "lm-lora__${model_name}__${dataset_name}"
        --wandb-mode "$wandb_mode"
        "${train_template_args[@]}"
        "${validation_template_args[@]}"
      )
      [[ -n "$wandb_entity" ]] && command+=(--wandb-entity "$wandb_entity")
      [[ "$dry_run" == "1" ]] && command+=(--dry-run --print-resolved-config)
      "${command[@]}"
    done
  done
done
