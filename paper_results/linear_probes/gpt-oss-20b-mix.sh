# Evaluate probes
model_name="gpt-oss-20b"
probe_dirs=(
    "/home/sinhan-yang/rethinking-calibration/outputs/gsm8k-train__gpt-oss-20b__A40--merged--triviaqa-train__gpt-oss-20b__A40/probes/linear__loss-bce__optimizer-sgd__lr-1e-3__lr_scheduler-none__weight_decay-1e-2__batch_size-32__epochs-100__preprocess-true__apply_sigmoid-false__num_samples-14946/layer_mean_pooled/best_probe.pt"
)
eval_datasets=(
    "aime-test"
)
for probe_dir in "${probe_dirs[@]}"; do
    train_dataset=$(echo "${probe_dir}" | sed "s|^outputs/||;s|__${model_name}/.*||")
    echo "Training dataset: ${train_dataset}"
    for eval_dataset in "${eval_datasets[@]}"; do
        echo "Evaluating on ${eval_dataset}"
        uv run python scripts/eval_probe.py \
            --eval_dir "neurips2026/outputs/${eval_dataset}__${model_name}/probing_dataset_cc" \
            --probe_path "${probe_dir}" \
            --output_dir "neurips2026/estimator_results/linear_probe/${model_name}__trained-on-mix-triviaqa-gsm8k/${eval_dataset}"
    done
done


# gpt-oss 