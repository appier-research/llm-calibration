# Construct probing datasets
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

# Train probes
probe_type="linear"
loss="bce"
optimizer="sgd"
lr_scheduler="none"
batch_size="32"
epochs="100"
num_samples=76523
train_folder="outputs/triviaqa-train__Qwen3-1_7B-non-thinking"
valid_folder="outputs/triviaqa-validation__Qwen3-1_7B-non-thinking"
probing_dataset_name="probing_dataset"
apply_sigmoid="false"

lr="2e-4"
weight_decay="1e-2"
preprocess="false"
echo "Training with lr=${lr}, weight_decay=${weight_decay}, preprocess=${preprocess}"
uv run python scripts/train_probe.py \
    --train_dir "${train_folder}/${probing_dataset_name}" \
    --val_dir "${valid_folder}/${probing_dataset_name}" \
    --probe_type ${probe_type} \
    --loss ${loss} \
    --optimizer ${optimizer} \
    --lr ${lr} \
    --lr_scheduler ${lr_scheduler} \
    --weight_decay ${weight_decay} \
    --batch_size ${batch_size} \
    --epochs ${epochs} \
    --num_samples ${num_samples} \
    --patience 100 \
    --wandb_project probe-training-scaling \
    --no_apply_sigmoid \
    ${preprocess_flag} \
    --layer_indices "" \
    --output_dir "${train_folder}/probes/${probe_type}__loss-${loss}__optimizer-${optimizer}__lr-${lr}__lr_scheduler-${lr_scheduler}__weight_decay-${weight_decay}__batch_size-${batch_size}__epochs-${epochs}__preprocess-${preprocess}__apply_sigmoid-${apply_sigmoid}__num_samples-${num_samples}"

# Example name: linear__loss-bce__optimizer-sgd__lr-1e-4__lr_scheduler-none__weight_decay-1e-4__batch_size-32__epochs-100__preprocess-false__apply_sigmoid-false__num_samples-76523