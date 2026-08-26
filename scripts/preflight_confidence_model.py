#!/usr/bin/env python3
"""Strategy-aware meta-device preflight for confidence-model backbones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.confidence_training.artifacts import atomic_write_json
from src.confidence_training.cli import print_json
from src.confidence_training.config import FULL_WEIGHT_MODEL_SPECS, RunConfig, lora_target_profile
from src.confidence_training.models import (
    LLMScoreModel,
    apply_lora,
    build_optimizer,
    parameter_report,
    prepare_full_weight_backbone,
)


EXPECTED_MODEL_TYPES = {
    "Olmo-3-7B-Instruct": "olmo3",
    "Qwen3-8B-non-thinking": "qwen3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON with a models list")
    parser.add_argument("--strategy", choices=("lora", "full_weight"), default="full_weight")
    parser.add_argument("--model", action="append", nargs=2, metavar=("MODEL_NAME", "MODEL_ID"), default=[])
    parser.add_argument("--model-name")
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--dataset-dir", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-resolved-config", action="store_true")
    return parser.parse_args()


def _model_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    if args.config:
        models.extend(json.loads(Path(args.config).read_text())["models"])
    models.extend(
        {"model_name": name, "model_id": model_id, "revision": args.revision}
        for name, model_id in args.model
    )
    if args.model_name or args.model_id:
        if not args.model_name or not args.model_id:
            raise ValueError("--model-name and --model-id must be supplied together")
        models.append(
            {"model_name": args.model_name, "model_id": args.model_id, "revision": args.revision}
        )
    if not models:
        raise ValueError("Provide --config, --model, or --model-name/--model-id")
    return models


def main() -> None:
    args = parse_args()
    models = _model_specs(args)
    resolved = {
        "preflight_version": 3,
        "strategy": args.strategy,
        "models": models,
        "dataset_dirs": args.dataset_dir,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "output": args.output,
    }
    if args.print_resolved_config or args.dry_run:
        print_json(resolved)
    if args.dry_run:
        return
    for folder in args.dataset_dir:
        for filename in ("ground_truth.jsonl", "sampled.jsonl"):
            path = Path(folder) / filename
            if not path.is_file():
                raise FileNotFoundError(path)

    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModel

    reports = []
    for spec in models:
        configuration = AutoConfig.from_pretrained(
            spec["model_id"], revision=spec.get("revision"), trust_remote_code=True
        )
        actual_model_type = str(getattr(configuration, "model_type", ""))
        expected_model_type = EXPECTED_MODEL_TYPES.get(spec["model_name"])
        if expected_model_type is not None and actual_model_type != expected_model_type:
            raise ValueError(
                f"Pinned model type drift for {spec['model_name']}: "
                f"{actual_model_type!r} != {expected_model_type!r}"
            )
        with init_empty_weights():
            backbone = AutoModel.from_config(
                configuration, trust_remote_code=True, dtype=torch.bfloat16
            )
            if args.strategy == "full_weight":
                pinned = FULL_WEIGHT_MODEL_SPECS.get(spec["model_name"])
                expected_count = int(pinned["backbone_parameters"]) if pinned else None
                prepare_full_weight_backbone(
                    backbone, expected_parameters=expected_count, enforce_bf16=True
                )
                matches = None
            else:
                modules, parameters = lora_target_profile(spec["model_name"])
                lora_config = RunConfig(
                    adaptation_strategy="lora",
                    model_id=spec["model_id"],
                    model_name=spec["model_name"],
                    model_revision=spec.get("revision"),
                    learning_rate=args.learning_rate,
                    batch_size=args.batch_size,
                    lora_target_modules=modules,
                    lora_target_parameters=parameters,
                )
                lora_config.validate(require_paths=False)
                backbone, matches = apply_lora(backbone, lora_config)
            score_model = LLMScoreModel(backbone)
        if not hasattr(backbone, "gradient_checkpointing_enable"):
            raise ValueError(f"{spec['model_name']} lacks gradient-checkpoint support")
        run_config = RunConfig(
            adaptation_strategy=args.strategy,
            model_id=spec["model_id"],
            model_name=spec["model_name"],
            model_revision=spec.get("revision"),
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_checkpointing=True,
        )
        run_config.validate(require_paths=False)
        optimizer = build_optimizer(score_model, run_config)
        report = parameter_report(
            score_model, optimizer, adaptation_strategy=args.strategy
        )
        if not report["optimizer_matches_trainable"]:
            raise RuntimeError("Meta optimizer membership validation failed")
        if not report["full_weight_trainable_backbone_matches_logical"]:
            raise RuntimeError("Meta full-weight backbone validation failed")
        reports.append(
            {
                **spec,
                "model_type": actual_model_type,
                "gradient_checkpointing_supported": True,
                "matched_targets": matches,
                "parameter_report": report,
            }
        )
    result = {**resolved, "ok": True, "models": reports}
    if args.output:
        atomic_write_json(args.output, result)
    print_json(result)


if __name__ == "__main__":
    main()
