# Full-Weight Confidence Fine-Tuning on a RunPod H200

## Purpose

Prepare, test, and validate the repository for a fixed-configuration preliminary
full-weight confidence-training experiment on one NVIDIA H200. Do not launch the
real 18-arm experiment as part of this preparation task. Stop after the code,
tests, dry runs, dataset checks, and two real-model H200 capacity/checkpoint
smokes pass, then report readiness and the measured smoke results.

This plan intentionally excludes GPT-OSS-20B. Native GPT-OSS MXFP4 experts are
not trainable, and dequantized BF16 GPT-OSS-20B plus gradients and PyTorch AdamW
state cannot fit on one 141 GB H200 under the fixed protocol. GPT-OSS full-weight
training will be designed separately.

Preserve all existing linear-probe and LoRA results. Do not interrupt an active
local LoRA controller, modify unrelated dirty files, delete artifacts, or launch
any full-weight arm without explicit follow-up authorization.

## Locked scientific design

### Research comparison

The adaptation strategy is `full_weight`: train every parameter of a separate
confidence-model copy of the source LM, plus its scalar regression head. The
copy predicts the fixed source model's saved expected accuracy for each query;
fine-tuning must never regenerate or alter the labels.

The preliminary matrix contains 18 optimization arms:

- Models:
  - `Olmo-3-7B-Instruct`
  - `Qwen3-8B-non-thinking`
- Datasets:
  - `triviaqa`
  - `math`
  - `mmlu`
- Training losses:
  - `mse`
  - `bce`
  - `ranknet`

Each optimization trajectory simultaneously tracks two raw-validation
checkpoint-selection policies:

- Lowest MSE, with strict improvement and earliest epoch retained on ties.
- Highest Spearman correlation, with strict improvement and earliest epoch
  retained on ties.

Thus the run produces 18 training trajectories and up to 36 selected
policy-level results. A single epoch may serve both policies and must be stored
only once.

### Pinned models and data folders

Use the existing pinned revisions:

| Model name | Hugging Face model ID | Revision |
|---|---|---|
| `Olmo-3-7B-Instruct` | `allenai/Olmo-3-7B-Instruct` | `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` |
| `Qwen3-8B-non-thinking` | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |

For each model, use these logical train/validation mappings:

| Dataset | Training folder | Validation folder | Expected counts |
|---|---|---|---:|
| TriviaQA | `outputs/triviaqa-train__<model>` | `outputs/triviaqa-validation__<model>` | 7,803 / 1,000 |
| MATH | `outputs/math-train__<model>` | `outputs/math-valid__<model>` | 11,498 / 500 |
| MMLU | `outputs/mmlu-validation__<model>` | `outputs/mmlu-test__<model>` | 1,531 / 1,000 |

`ground_truth.jsonl` is authoritative. Use the first prompt for each required
`example_id` in `sampled.jsonl`. Full-weight training must not read or require
`probing_dataset_cc` hidden states.

Before any smoke or training run, audit all 12 model/dataset/split folders in a
new adapted-LM/prompt-only audit mode. This mode must not require
`probing_dataset_cc`. The audit must verify counts from `ground_truth.jsonl`,
unique and complete prompt joins, valid `num_samples` and `num_correct`, exact
expected-accuracy fractions, cross-model equality of ordered IDs and canonical
question text for each logical split, and train/validation disjointness by both
ID and canonical question. Fail rather than repair data on the RunPod machine.
Record the machine-readable audit report and hashes.

### Prompt contract

Use the same shared data and chat-template path as the LoRA experiment:

- Render the first stored prompt with the pinned tokenizer.
- Set `add_generation_prompt=True`.
- Use no truncation.
- Fail if any rendered prompt exceeds the model context limit.
- Preserve model-specific tokenizer/chat-template behavior.
- Hash ordered rendered prompts separately for train and validation.
- Use the final non-padding token based on attention-mask positions, supporting
  either padding side.

Do not introduce a full-weight-specific prompt, suffix, answer token, or target.

### Fixed optimization configuration

Use exactly this configuration for all 18 preliminary arms:

| Field | Value |
|---|---|
| Adaptation strategy | `full_weight` |
| Batch size | `16` |
| Learning rate | `1e-5` |
| Epochs | `3` |
| Seed | `42` |
| Optimizer | PyTorch `torch.optim.AdamW` |
| AdamW betas | `(0.9, 0.999)` |
| AdamW epsilon | `1e-8` |
| Weight decay | `0.01` on every trainable parameter |
| Gradient clipping | Global trainable-parameter norm `1.0` |
| Scheduler | None |
| Warmup | None |
| Gradient accumulation | None |
| Gradient checkpointing | Enabled, non-reentrant |
| Training processes | Exactly one |
| Data-loader workers | `0` |
| Training order | Deterministic shuffle using `seed + epoch` |
| Validation order | Fixed |
| `drop_last` | `False` |
| Training-stat cadence | Every 10 optimizer steps and the final partial interval |
| Resume state | Disabled for this preliminary run |

The backbone forward/backward path and stored backbone weights use BF16. The
regression head, pooled representation passed to the head, targets, score loss,
loss reduction, saved validation scores, and metrics use FP32. One LR applies to
both the head and backbone. Do not create no-decay parameter groups: biases and
normalization parameters receive weight decay too.

Loss definitions remain:

- MSE: `MSE(sigmoid(raw_score), expected_accuracy)`.
- BCE: `BCEWithLogits(raw_score, expected_accuracy)`, including fractional
  targets.
- RankNet: all distinct-target pairs within each physical batch, operating on
  raw scores, temperature `1.0`, tie weight `0.0`. A batch of 16 therefore has up
  to 120 pairs. Do not replace the physical batch or emulate it with gradient
  accumulation.

### Validation and checkpoint contract

- Validate at step zero, log it, and exclude it from selection.
- Thereafter validate only at epoch boundaries.
- Compute raw probabilities once with `sigmoid(raw_score)` and calculate metrics
  from the shared `src/metrics` implementation.
- MSE is primary and Spearman is secondary in reporting, but track both selectors
  independently online.
- Retain at most the two unique full-model epochs currently referenced by the
  selectors. Delete only superseded, run-owned checkpoint weights after the new
  checkpoint and selector manifest are durably written.
- Each retained checkpoint contains the complete BF16 Hugging Face backbone, the
  FP32 regression head, validation IDs, FP32 scores/probabilities/targets,
  metrics, and hashes.
- Run configuration, history, selectors, provenance, tokenizer, and status must
  remain independently available even if weights are later pruned explicitly.
- Do not fit isotonic regression during these preliminary training runs.

## Required implementation work

### 1. Add an adapted-LM data-audit mode

The current `audit_confidence_data.py` always audits probing metadata, targets,
and hidden states. Add an explicit audit scope for adapted-LM training that
checks only authoritative ground truth and stored prompts while preserving the
existing strict probing-artifact scope for linear experiments.

- Make the scope explicit in CLI/config output; do not infer it from missing
  files.
- In adapted-LM scope, require only `ground_truth.jsonl` and `sampled.jsonl` and
  perform every count, target, prompt-join, cross-model, and cross-split check
  described above.
- Keep `ground_truth.jsonl` authoritative.
- Emit ordered data/prompt hashes and the selected audit scope in the report.
- Add a 12-entry OLMo/Qwen matrix at
  `paper_results/confidence_training/full_weight_data_matrix.json` rather than
  reusing the 18-entry matrix that includes GPT-OSS.
- Do not copy, reconstruct, or audit probing hidden states on RunPod for this
  experiment.

### 2. Harden true full-weight model construction

Update the unified model path rather than creating an unrelated trainer.

- Load OLMo and Qwen backbones at the pinned revision in BF16.
- Set `use_cache=False` and enable non-reentrant gradient checkpointing.
- Explicitly set every logical backbone parameter to `requires_grad=True` for
  `full_weight`.
- Keep the regression head in FP32 and train it.
- Reject any quantized or otherwise frozen logical backbone parameter in
  `full_weight` mode. In particular, fail fast if GPT-OSS/MXFP4 is accidentally
  supplied; do not silently run partial-backbone training.
- Expand the parameter report to prove:
  - exact total and trainable counts;
  - head/backbone counts;
  - dtype counts;
  - every trainable parameter appears exactly once in the optimizer;
  - no optimizer parameter is frozen;
  - full-weight trainable backbone count equals the logical backbone count.
- Persist the report and actual dtypes in `resolved_config.json` and
  `run_manifest.json`.

The expected meta-device backbone counts are currently:

- OLMo: `6,887,272,448`
- Qwen: `7,568,405,504`

Treat count drift as a failed preflight unless it is deliberately reviewed and
the pinned revision or implementation changed.

### 3. Extend preflight and capacity-smoke tooling

Generalize or replace the LoRA-only smoke utility with a strategy-aware
full-weight utility. It must be shell-friendly and accept model, revision,
dataset folders, batch size, LR, output path, and dry-run/config printing.

Meta-device preflight must validate model type, parameter counts, dtypes,
gradient-checkpoint support, optimizer membership, and absence of quantized or
frozen logical weights without allocating full model storage.

The real H200 capacity smoke must:

1. Tokenize every prompt across all three datasets for the selected model.
2. Construct a deterministic worst-case batch of 16 from the longest rendered
   prompts, rather than a random batch.
3. Load the real BF16 full-weight model on the H200.
4. Run forward, FP32 loss, backward, global clipping, and the first AdamW step.
   The optimizer step is mandatory because AdamW state is allocated lazily.
5. Verify finite score, loss, and gradient statistics.
6. Verify at least one backbone parameter and the regression head changed.
7. Record allocated and reserved peak VRAM after synchronization, batch token
   lengths, wall time, parameter report, CUDA/PyTorch/Transformers versions, GPU
   identity, and free memory.
8. Save a full checkpoint, release the original model and optimizer, reload the
   checkpoint, and compare raw scores and probabilities on a fixed batch within
   an explicit BF16 tolerance.
9. Verify checkpoint contents and hashes and report checkpoint bytes and save/
   reload time.

Run one real capacity/checkpoint smoke for OLMo and one for Qwen. Both must pass
before the 18-arm launcher is authorized.

### 4. Add the fixed preliminary launcher

Create:

`paper_results/confidence_training/run_preliminary_full_weight.sh`

It must:

- Use `set -euo pipefail`.
- Iterate sequentially over two models, three datasets, and three losses.
- Invoke exactly one `uv run python scripts/train_confidence_model.py` process at
  a time.
- Pin the fixed protocol values and reject environment overrides that conflict
  with batch size 16, LR `1e-5`, epochs 3, or seed 42.
- Support `MODEL_FILTER`, `DATASET_FILTER`, `LOSS_FILTER`, `OUTPUT_ROOT`,
  `WANDB_ENTITY`, `WANDB_MODE`, and `DRY_RUN` for testing and recovery without
  source edits.
- Default the output root to
  `outputs/confidence_training/preliminary_full_weight/seed-42`.
- Name runs as
  `<dataset>__<model>__full_weight__<loss>__lr-1e-5__bs-16__seed-42`.
- Skip only manifests whose status is exactly `complete`.
- Refuse nonempty/incomplete destinations instead of overwriting them, since
  rolling full-weight resume state is disabled.
- Check required input files and available output space before loading a model.
- Pass `--gradient-checkpointing`, `--no-standardize`,
  `--log-every-steps 10`, and `--no-save-resume-state` explicitly.
- Support `DRY_RUN=1` using `--dry-run --print-resolved-config` without loading a
  model, initializing W&B, or writing run outputs.
- Exclude GPT-OSS unconditionally and fail if a filter attempts to select it.

### 5. Configure W&B and local logging

Use direct online W&B logging on RunPod with entity `brian-wu`. Local artifacts
remain authoritative.

Create six projects:

`lm-full-weight__<model_name>__<dataset_name>`

Each project contains the MSE, BCE, and RankNet runs for that model/dataset.
Use deterministic run IDs and resumable W&B identity, but do not interpret W&B
resume as scientific checkpoint resume. Log:

- interval training loss, mean unclipped gradient norm, interval examples and
  seconds, optimizer steps, epoch step, and LR every 10 steps;
- step-zero and epoch-boundary raw validation metrics;
- epoch training loss, elapsed time, and validation metrics;
- final selector epochs/values, run status, parameter/dtype report, peak VRAM,
  and artifact paths/hashes in the summary.

Retain support for offline history backfill if network logging fails. Verify that
backfill accepts `adaptation_strategy=full_weight` and does not duplicate a run.

### 6. Validate full-weight artifacts and summaries

- Verify full-weight save/reload independently of the real capacity smoke using
  tiny fake Hugging Face-style backbones.
- Ensure loading a selected checkpoint does not fall back to the original base
  revision.
- Ensure two selectors can share one full checkpoint and that superseded full
  checkpoints are removed only after a replacement is safely committed.
- Ensure the summary utility accepts `--expected-runs 18`, rejects missing or
  failed arms, and reports both selection policies plus per-epoch and total wall
  time.
- Produce final JSON and CSV summaries under the preliminary full-weight output
  root only after all 18 runs complete.

## Test plan

All automated tests must be CPU-only and network-free and run with:

```bash
uv run pytest
```

Add or extend tests under `tests/` for the following.

### Configuration and CLI tests

- Verify adapted-LM audit scope succeeds without probing artifacts, while the
  existing probing scope remains strict and reports missing probing artifacts.
- Verify the 12-entry full-weight data matrix excludes GPT-OSS and exercises all
  expected model/dataset/split folders.
- Resolve all combinations of `full_weight` with MSE/BCE/RankNet.
- Verify JSON defaults and CLI override precedence.
- Verify the locked batch size, LR, epochs, seed, optimizer protocol, gradient
  checkpointing, and logging cadence in all 18 launcher dry runs.
- Reject standardization, LoRA-only arguments, GPT-OSS, quantized full-weight
  modules, missing folders, invalid filters, and conflicting fixed overrides.
- Verify dry-run has no filesystem or W&B side effects.

### Model, precision, and optimizer tests

- Every fake backbone parameter and the FP32 head are trainable in full-weight
  mode.
- Backbone parameters are BF16 and head/loss/targets are FP32.
- PyTorch AdamW uses one shared LR, fixed betas/epsilon/weight decay, includes all
  and only trainable parameters exactly once, and applies decay to biases and
  normalization parameters.
- Global clipping covers every trainable parameter.
- A deliberately frozen or quantized logical backbone parameter causes a hard
  failure.
- Parameter reports distinguish logical, trainable, optimizer, component, and
  dtype counts accurately.

### Training and loss tests

- Preserve existing MSE, fractional-BCE, and RankNet semantics.
- RankNet uses all distinct-target pairs in the physical batch, ignores ties,
  and handles an incomplete final batch.
- Deterministic epoch shuffling and fixed validation order remain intact.
- Step logging occurs at the requested cadence and at the final partial interval.
- Step-zero validation is logged but never selected.
- Epoch-only validation and independent dual selectors behave correctly.

### Checkpoint and artifact tests

- Full backbone plus FP32 head save/reload round-trips scores.
- Validation tensors and ordered IDs survive round-trip in FP32.
- MSE and Spearman can select different epochs or share one epoch.
- Strict ties retain the earliest epoch.
- At most two referenced full checkpoints remain, with atomic replacement and
  safe cleanup behavior.
- Nonempty output protection, immutable scientific configuration, hashes, and
  failure manifests work for full-weight runs.
- Resume-state files are absent under the fixed preliminary protocol.

### Launcher, W&B, and summary tests

- `bash -n` passes for the launcher.
- A launcher dry run emits exactly 18 unique commands/configurations in the
  expected deterministic order.
- Filters select the exact intended subset.
- Project names produce exactly six W&B projects and deterministic run IDs.
- A completed synthetic full-weight history can be backfilled once without
  duplication.
- A tiny end-to-end synthetic 18-arm tree passes the summary integrity checks;
  missing, duplicate, mismatched, or failed runs are rejected.

## Required execution gates

The preparation agent must complete these gates in order and stop before real
training:

1. Inspect the dirty worktree and preserve unrelated user changes.
2. Implement the full-weight hardening, preflight/smoke utility, launcher, W&B
   coverage, and tests.
3. Run `uv run pytest` and report the exact pass count.
4. Run `bash -n paper_results/confidence_training/run_preliminary_full_weight.sh`.
5. Run the complete 18-arm launcher dry run and mechanically validate all fixed
   fields, paths, revisions, names, projects, and uniqueness.
6. On the RunPod H200, record environment and GPU information and rerun the
   adapted-LM-scope data audit against the transferred inputs.
7. Run the OLMo worst-case batch-16 full-weight capacity/checkpoint smoke.
8. Release all model/optimizer memory and run the Qwen worst-case batch-16
   full-weight capacity/checkpoint smoke.
9. Confirm each smoke leaves safe VRAM headroom for sequence-length variability
   and checkpointing. Do not infer capacity merely because model loading or a
   forward pass succeeds.
10. Report readiness, audit hashes, smoke peak memory/timing, checkpoint sizes and
    reload error, test results, dry-run matrix, and any remaining risk.
11. Wait for explicit authorization before starting the sequential 18-arm
    launcher.

If either real-model smoke OOMs or fails reload equivalence, stop and report the
evidence. Do not silently reduce batch size, add gradient accumulation, change
precision or optimizer, truncate prompts, freeze parameters, or start a subset
of the experiment.

## Acceptance criteria

Preparation is complete only when all of the following are proven:

- The resolved experiment matrix contains exactly 18 OLMo/Qwen full-weight arms
  and no GPT-OSS arm.
- Every fixed scientific field matches this document.
- All 12 input folders pass the strict data audit on RunPod and their hashes are
  recorded.
- `uv run pytest` passes in full.
- Launcher syntax and the complete dry-run matrix pass.
- Both actual H200 full-weight capacity/checkpoint smokes pass through the first
  AdamW step and checkpoint reload with safe VRAM headroom.
- W&B configuration resolves to six projects and the local history remains
  sufficient for deterministic backfill.
- No real preliminary full-weight arm has been launched during preparation.

