#!/usr/bin/env python3
"""Run a worst-case H200 full-weight step and verify a complete checkpoint reload."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import torch

from src.confidence_training.artifacts import atomic_write_json, sha256_path
from src.confidence_training.collate import PromptCollator
from src.confidence_training.config import FULL_WEIGHT_MODEL_SPECS, RunConfig
from src.confidence_training.data import load_confidence_examples, render_prompts
from src.confidence_training.losses import confidence_loss
from src.confidence_training.models import (
    build_optimizer,
    load_llm_score_model,
    load_score_model,
    parameter_report,
    save_score_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", choices=tuple(FULL_WEIGHT_MODEL_SPECS), required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--dataset-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reload-atol", type=float, default=1e-3)
    parser.add_argument("--reload-rtol", type=float, default=1e-3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-resolved-config", action="store_true")
    return parser.parse_args()


def _checkpoint_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _changed_sample(parameter: torch.nn.Parameter) -> tuple[torch.Tensor, torch.Tensor]:
    assert parameter.grad is not None
    gradient = parameter.grad.detach().reshape(-1)
    finite_nonzero = torch.nonzero(torch.isfinite(gradient) & (gradient != 0), as_tuple=False).flatten()
    if not len(finite_nonzero):
        raise RuntimeError("Selected backbone gradient has no finite nonzero elements")
    indices = finite_nonzero[:1024]
    return indices, parameter.detach().reshape(-1)[indices].float().cpu().clone()


def main() -> None:
    args = parse_args()
    pinned = FULL_WEIGHT_MODEL_SPECS[args.model_name]
    model_id = args.model_id or str(pinned["model_id"])
    revision = args.revision or str(pinned["revision"])
    resolved = {
        "smoke_version": 1,
        "adaptation_strategy": "full_weight",
        "model_name": args.model_name,
        "model_id": model_id,
        "revision": revision,
        "dataset_dirs": args.dataset_dir,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "reload_atol": args.reload_atol,
        "reload_rtol": args.reload_rtol,
    }
    if args.batch_size != 16 or args.learning_rate != 1e-5 or args.seed != 42:
        raise ValueError("Smoke protocol fixes batch_size=16, learning_rate=1e-5, seed=42")
    if len(args.dataset_dir) != 6:
        raise ValueError("Provide exactly six train/validation dataset folders for one model")
    if args.print_resolved_config or args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for the real full-weight smoke")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Smoke output directory is nonempty: {output}")
    for folder in args.dataset_dir:
        for filename in ("ground_truth.jsonl", "sampled.jsonl"):
            if not (Path(folder) / filename).is_file():
                raise FileNotFoundError(Path(folder) / filename)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "resolved_config.json", resolved)

    from transformers import AutoTokenizer, __version__ as transformers_version

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    ranked: list[tuple[int, str, int, Any]] = []
    all_lengths: list[int] = []
    for folder_index, folder in enumerate(args.dataset_dir):
        examples = load_confidence_examples(folder)
        for start in range(0, len(examples), 256):
            chunk = examples[start : start + 256]
            rendered = render_prompts(tokenizer, chunk)
            encoded = tokenizer(rendered, padding=False, truncation=False, add_special_tokens=False)
            for offset, tokens in enumerate(encoded["input_ids"]):
                length = len(tokens)
                limit = getattr(tokenizer, "model_max_length", None)
                if isinstance(limit, int) and limit < 10**9 and length > limit:
                    raise ValueError(f"Prompt length {length} exceeds model context limit {limit}")
                example = chunk[offset]
                ranked.append((length, str(folder), start + offset, example))
                all_lengths.append(length)
    ranked.sort(key=lambda row: (-row[0], row[1], row[2], row[3].example_id))
    selected_rows = ranked[: args.batch_size]
    selected = [row[3] for row in selected_rows]

    config = RunConfig(
        run_name=f"capacity-smoke__{args.model_name}",
        dataset_name="all-three-datasets",
        model_name=args.model_name,
        model_id=model_id,
        model_revision=revision,
        adaptation_strategy="full_weight",
        loss="mse",
        train_dir=args.dataset_dir[0],
        validation_dir=args.dataset_dir[1],
        output_dir=str(output),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=1,
        seed=args.seed,
        gradient_checkpointing=True,
        num_workers=0,
        log_every_steps=10,
        save_resume_state=False,
    )
    config.validate()
    started = time.perf_counter()
    model, loaded_tokenizer, matches = load_llm_score_model(config)
    assert matches is None
    loaded_tokenizer.padding_side = "right"
    loaded_tokenizer.save_pretrained(output / "tokenizer")
    device = torch.device("cuda")
    model.to(device)
    optimizer = build_optimizer(model, config)
    report = parameter_report(model, optimizer, adaptation_strategy="full_weight")
    if not report["optimizer_matches_trainable"] or not report[
        "full_weight_trainable_backbone_matches_logical"
    ]:
        raise RuntimeError("Full-weight parameter/optimizer report failed")
    collator = PromptCollator(loaded_tokenizer, device=device, max_sequence_length=None)
    batch = collator(selected)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    step_started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scores = model.forward_scores(**batch["inputs"]).float()
    loss = confidence_loss(scores, batch["targets"].float(), "mse")
    if not torch.isfinite(scores).all() or not torch.isfinite(loss):
        raise RuntimeError("Smoke produced non-finite scores or loss")
    loss.backward()
    backbone_parameter = None
    for candidate in reversed(list(model.backbone.parameters())):
        if candidate.grad is None:
            continue
        try:
            sample_indices, backbone_before = _changed_sample(candidate)
        except RuntimeError:
            continue
        backbone_parameter = candidate
        break
    if backbone_parameter is None:
        raise RuntimeError("No backbone parameter has a finite nonzero gradient")
    head_before = {
        name: value.detach().cpu().clone() for name, value in model.regression_head.state_dict().items()
    }
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
    )
    if not math.isfinite(float(gradient_norm)):
        raise RuntimeError("Smoke produced a non-finite gradient norm")
    optimizer.step()
    torch.cuda.synchronize()
    step_seconds = time.perf_counter() - step_started
    backbone_after = backbone_parameter.detach().reshape(-1)[sample_indices].float().cpu()
    backbone_changed = not torch.equal(backbone_before, backbone_after)
    head_changed = any(
        not torch.equal(head_before[name], value.detach().cpu())
        for name, value in model.regression_head.state_dict().items()
    )
    if not backbone_changed or not head_changed:
        raise RuntimeError(
            f"Optimizer update check failed: backbone_changed={backbone_changed}, head_changed={head_changed}"
        )
    model.eval()
    fixed_examples = selected[:2]
    fixed_batch = collator(fixed_examples)
    with torch.inference_mode():
        reference_scores = model.forward_scores(**fixed_batch["inputs"]).float().cpu()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    free_memory, total_memory = torch.cuda.mem_get_info()
    loss_value = float(loss.detach().cpu())
    score_min = float(scores.detach().min().cpu())
    score_max = float(scores.detach().max().cpu())
    gradient_norm_value = float(gradient_norm.detach().cpu())

    checkpoint = output / "checkpoint" / "model"
    save_started = time.perf_counter()
    save_score_model(model, checkpoint, config)
    save_seconds = time.perf_counter() - save_started
    checkpoint_hash = sha256_path(checkpoint)
    checkpoint_bytes = _checkpoint_bytes(checkpoint)
    required_checkpoint_paths = (
        checkpoint / "backbone",
        checkpoint / "regression_head.pt",
        checkpoint / "model.json",
    )
    if not all(path.exists() for path in required_checkpoint_paths):
        raise RuntimeError("Full-weight checkpoint contents are incomplete")

    del optimizer, model, batch, fixed_batch, backbone_parameter, loss, scores, gradient_norm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    reload_started = time.perf_counter()
    restored, restored_tokenizer = load_score_model(checkpoint, config, device=device)
    reload_seconds = time.perf_counter() - reload_started
    restored_tokenizer.padding_side = "right"
    restored_batch = PromptCollator(
        restored_tokenizer, device=device, max_sequence_length=None
    )(fixed_examples)
    restored.eval()
    with torch.inference_mode():
        restored_scores = restored.forward_scores(**restored_batch["inputs"]).float().cpu()
    restored_probabilities = torch.sigmoid(restored_scores)
    reference_probabilities = torch.sigmoid(reference_scores)
    torch.testing.assert_close(
        restored_scores, reference_scores, atol=args.reload_atol, rtol=args.reload_rtol
    )
    torch.testing.assert_close(
        restored_probabilities,
        reference_probabilities,
        atol=args.reload_atol,
        rtol=args.reload_rtol,
    )
    properties = torch.cuda.get_device_properties(device)
    result = {
        **resolved,
        "ok": True,
        "selected_batch": [
            {"tokens": row[0], "folder": row[1], "index": row[2], "example_id": row[3].example_id}
            for row in selected_rows
        ],
        "all_prompt_count": len(all_lengths),
        "maximum_dataset_tokens": max(all_lengths),
        "loss": loss_value,
        "score_min": score_min,
        "score_max": score_max,
        "gradient_norm": gradient_norm_value,
        "backbone_changed": backbone_changed,
        "head_changed": head_changed,
        "step_seconds": step_seconds,
        "save_seconds": save_seconds,
        "reload_seconds": reload_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "free_memory_after_step_bytes": free_memory,
        "total_gpu_memory_bytes": total_memory,
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_reload_max_score_abs_error": float((restored_scores - reference_scores).abs().max()),
        "checkpoint_reload_max_probability_abs_error": float(
            (restored_probabilities - reference_probabilities).abs().max()
        ),
        "parameter_report": report,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers_version,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
        },
    }
    atomic_write_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
