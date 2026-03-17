# Construct probing datasets
model="Qwen/Qwen3-8B"
model_name="Qwen3-8B-non-thinking"
prefixes=(
    # Evaluation datasets
    "triviaqa-validation"
    "simpleqa-verified"
    "gsm8k-test"
    "math-500"
    "aime25-test"
    "mmlu-validation"
    "gpqa-diamond"
    # Training datasets
    "triviaqa-train"
    "gsm8k-train"
    "math-train"
    "math-valid"
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

# Train linear probes
probe_type="linear"
loss="bce"
optimizer="sgd"
lr_scheduler="none"
batch_size="32"
epochs="100"
probing_dataset_name="probing_dataset"
apply_sigmoid="false"
weight_decay="1e-2"

model_name="Qwen3-8B-non-thinking"
num_samples_l=(76523 7473 11498)
train_datasets=("triviaqa-train" "gsm8k-train" "math-train")
valid_datasets=("triviaqa-validation" "gsm8k-test" "math-valid")

lrs=("1e-5" "2e-5" "5e-5" "1e-4" "2e-4" "5e-4")
preprocesses=("true" "false")

for i in "${!num_samples_l[@]}"; do
    num_samples=${num_samples_l[i]}
    train_folder="outputs/${train_datasets[i]}__${model_name}"
    valid_folder="outputs/${valid_datasets[i]}__${model_name}"
    for preprocess in "${preprocesses[@]}"; do
        if [[ "${preprocess}" == "true" ]]; then
            preprocess_flag="--preprocess"
        else
            preprocess_flag=""
        fi
        for lr in "${lrs[@]}"; do
            echo "Training with lr=${lr}, weight_decay=${weight_decay}, preprocess=${preprocess}, num_samples=${num_samples}, train_folder=${train_folder}, valid_folder=${valid_folder}"
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
        done
    done
done
# Example name: linear__loss-bce__optimizer-sgd__lr-1e-4__lr_scheduler-none__weight_decay-1e-4__batch_size-32__epochs-100__preprocess-false__apply_sigmoid-false__num_samples-76523

# Evaluate probes
model_name="Qwen3-8B-non-thinking"
probe_dirs=(
    "outputs/triviaqa-train__${model_name}/probes/linear__loss-bce__optimizer-sgd__lr-1e-4__lr_scheduler-none__weight_decay-1e-2__batch_size-32__epochs-100__preprocess-true__apply_sigmoid-false__num_samples-59784/layer_mean_pooled/best_probe.pt"
    "outputs/gsm8k-train__${model_name}/probes/linear__loss-bce__optimizer-sgd__lr-2e-5__lr_scheduler-none__weight_decay-1e-2__batch_size-32__epochs-500__preprocess-false__apply_sigmoid-false__num_samples-7473/layer_mean_pooled/best_probe.pt"
    "outputs/math-train__${model_name}/probes/linear__loss-bce__optimizer-sgd__lr-1e-4__lr_scheduler-none__weight_decay-1e-2__batch_size-32__epochs-500__preprocess-false__apply_sigmoid-false__num_samples-11498/layer_mean_pooled/best_probe.pt"
)
eval_datasets=(
    "triviaqa-validation"
    "simpleqa-verified"
    "gsm8k-test"
    "math-500"
    "aime25-test"
    "mmlu-validation"
    "gpqa-diamond"
)
for probe_dir in "${probe_dirs[@]}"; do
    train_dataset=$(echo "${probe_dir}" | sed "s|^outputs/||;s|__${model_name}/.*||")
    echo "Training dataset: ${train_dataset}"
    for eval_dataset in "${eval_datasets[@]}"; do
        echo "Evaluating on ${eval_dataset}"
        uv run python scripts/eval_probe.py \
            --eval_dir "outputs/${eval_dataset}__${model_name}/probing_dataset" \
            --probe_path "${probe_dir}" \
            --output_dir "estimator_results/linear_probe/${model_name}__trained-on-${train_dataset}/${eval_dataset}"
    done
done
