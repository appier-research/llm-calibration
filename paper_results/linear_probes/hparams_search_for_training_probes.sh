# Train probes
probe_type="linear"
loss="bce"
optimizer="sgd"
lr_scheduler="none"
batch_size="32"
epochs="500"
probing_dataset_name="probing_dataset"
apply_sigmoid="false"
weight_decay="1e-2"

model_name="Qwen3-8B-non-thinking"
num_samples_l=(7473 11498)
train_datasets=("gsm8k-train" "math-train")
valid_datasets=("gsm8k-test" "math-valid")

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