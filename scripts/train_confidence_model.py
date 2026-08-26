#!/usr/bin/env python3
"""Train one confidence model and retain simultaneous best-MSE/Spearman states."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from src.confidence_training.artifacts import RunArtifacts
from src.confidence_training.cli import add_run_arguments, config_from_namespace, print_json
from src.confidence_training.collate import PromptCollator, collate_linear
from src.confidence_training.data import (
    LinearFeatureDataset,
    PromptConfidenceDataset,
    ground_truth_hash,
    load_confidence_examples,
    prompt_hash,
    render_prompts,
    rendered_prompt_hash,
    tokenize_rendered_prompts,
)
from src.confidence_training.models import (
    LinearScoreModel,
    build_optimizer,
    load_llm_score_model,
    parameter_report,
)
from src.confidence_training.control import TrainingControl
from src.confidence_training.training import set_global_seed, train_run
from src.confidence_training.wandb_logging import WandbLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--rung", type=int, default=0)
    return parser.parse_args()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def preflight_prompts(tokenizer, examples, config, chat_template_kwargs) -> str:
    """Render/hash every prompt and check lengths before creating run artifacts."""

    rendered_all: list[str] = []
    preflight_batch_size = max(1, min(256, config.batch_size * 8))
    for start in range(0, len(examples), preflight_batch_size):
        rows = examples[start : start + preflight_batch_size]
        rendered = render_prompts(
            tokenizer,
            rows,
            chat_template_kwargs=chat_template_kwargs,
        )
        tokenize_rendered_prompts(
            tokenizer,
            rendered,
            device="cpu",
            max_sequence_length=config.max_sequence_length,
        )
        rendered_all.extend(rendered)
    return rendered_prompt_hash(
        [example.example_id for example in examples], rendered_all
    )


def main() -> None:
    args = parse_args()
    config = config_from_namespace(args)
    control = TrainingControl(stop_after_epoch=args.stop_after_epoch, rung=args.rung)
    control.validate(config)
    if args.print_resolved_config or args.dry_run:
        print_json({**config.to_dict(), "_training_control": control.to_dict()})
    if args.dry_run:
        return

    set_global_seed(config.seed)
    device = torch.device(
        config.device
        if torch.cuda.is_available() or not config.device.startswith("cuda")
        else "cpu"
    )
    examples_train = load_confidence_examples(config.train_dir)
    examples_validation = load_confidence_examples(config.validation_dir)
    feature_provenance: dict[str, Any] = {}
    rendered_hashes: dict[str, str | None] = {
        "train_rendered_prompt_hash": None,
        "validation_rendered_prompt_hash": None,
    }

    if config.adaptation_strategy == "linear_probe":
        expected_standardization = "gpt-oss" in config.model_name.casefold()
        if config.standardize != expected_standardization:
            raise ValueError(
                "Linear standardization policy requires enabled=true only for GPT-OSS "
                f"(model_name={config.model_name!r})"
            )
        train_dataset = LinearFeatureDataset.load(
            config.train_dir, config.probing_dataset_name
        )
        validation_dataset = LinearFeatureDataset.load(
            config.validation_dir, config.probing_dataset_name
        )
        expected_train_ids = [example.example_id for example in examples_train]
        expected_validation_ids = [example.example_id for example in examples_validation]
        if train_dataset.example_ids != expected_train_ids:
            raise ValueError("Training feature order does not match ground truth")
        if validation_dataset.example_ids != expected_validation_ids:
            raise ValueError("Validation feature order does not match ground truth")
        train_metadata = json.loads(
            (
                Path(config.train_dir)
                / config.probing_dataset_name
                / "metadata.json"
            ).read_text()
        )
        validation_metadata = json.loads(
            (
                Path(config.validation_dir)
                / config.probing_dataset_name
                / "metadata.json"
            ).read_text()
        )
        rendered_hashes = {
            "train_rendered_prompt_hash": train_metadata.get("rendered_prompt_hash"),
            "validation_rendered_prompt_hash": validation_metadata.get(
                "rendered_prompt_hash"
            ),
        }
        feature_provenance = {
            "train_feature_construction_hash": train_metadata.get("construction_hash"),
            "validation_feature_construction_hash": validation_metadata.get(
                "construction_hash"
            ),
            "pooling": "mean_across_saved_layers_after_last_non_padding_token_pooling",
        }
        model = LinearScoreModel(train_dataset.features.shape[1])
        if config.standardize:
            model.set_standardization(
                train_dataset.features, epsilon=config.standardization_epsilon
            )
        tokenizer = None
        lora_matches = None
        train_collate = validation_collate = collate_linear
    else:
        if config.standardize:
            raise ValueError("Feature standardization applies only to linear probes")
        model, tokenizer, lora_matches = load_llm_score_model(config)
        train_template_kwargs = (
            config.train_chat_template_kwargs
            if config.train_chat_template_kwargs is not None
            else config.chat_template_kwargs
        )
        validation_template_kwargs = (
            config.validation_chat_template_kwargs
            if config.validation_chat_template_kwargs is not None
            else config.chat_template_kwargs
        )
        rendered_hashes = {
            "train_rendered_prompt_hash": preflight_prompts(
                tokenizer, examples_train, config, train_template_kwargs
            ),
            "validation_rendered_prompt_hash": preflight_prompts(
                tokenizer, examples_validation, config, validation_template_kwargs
            ),
        }
        train_dataset = PromptConfidenceDataset(examples_train)
        validation_dataset = PromptConfidenceDataset(examples_validation)
        train_collate = PromptCollator(
            tokenizer,
            device=device,
            max_sequence_length=config.max_sequence_length,
            chat_template_kwargs=train_template_kwargs,
        )
        validation_collate = PromptCollator(
            tokenizer,
            device=device,
            max_sequence_length=config.max_sequence_length,
            chat_template_kwargs=validation_template_kwargs,
        )

    model.to(device)
    optimizer = build_optimizer(model, config)
    report = parameter_report(
        model, optimizer, adaptation_strategy=config.adaptation_strategy
    )
    if not report["optimizer_matches_trainable"]:
        raise RuntimeError("Optimizer membership differs from trainable parameters")
    if not report["full_weight_trainable_backbone_matches_logical"]:
        raise RuntimeError("Full-weight trainable backbone count is not logical count")

    provenance = {
        "git_commit": git_commit(),
        "train_ground_truth_hash": ground_truth_hash(
            [example.ground_truth for example in examples_train]
        ),
        "validation_ground_truth_hash": ground_truth_hash(
            [example.ground_truth for example in examples_validation]
        ),
        "train_prompt_hash": prompt_hash(examples_train),
        "validation_prompt_hash": prompt_hash(examples_validation),
        **rendered_hashes,
        **feature_provenance,
        "model_revision": getattr(getattr(model, "backbone", None), "config", None)
        and getattr(model.backbone.config, "_commit_hash", config.model_revision),
        "tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        if tokenizer is not None
        else None,
        "actual_device": str(device),
        "parameter_report": report,
        "lora_target_matches": lora_matches,
    }
    artifacts = RunArtifacts(config)
    artifacts.initialize(provenance)
    if tokenizer is not None and not (artifacts.root / "tokenizer").exists():
        tokenizer.save_pretrained(artifacts.root / "tokenizer")
    wandb_logger = WandbLogger.create(
        config,
        {**config.to_dict(), "_training_control": control.to_dict()},
    )
    callback = wandb_logger.log if wandb_logger is not None else None
    try:
        manifest = train_run(
            model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            train_collate=train_collate,
            validation_collate=validation_collate,
            config=config,
            artifacts=artifacts,
            log_callback=callback,
            control=control,
        )
        if wandb_logger is not None:
            wandb_logger.update_summary(manifest)
    except Exception as error:
        artifacts.mark_failed(str(error))
        if wandb_logger is not None and artifacts.manifest_path.exists():
            wandb_logger.update_summary(json.loads(artifacts.manifest_path.read_text()))
        raise
    finally:
        if wandb_logger is not None:
            wandb_logger.finish()
    print_json(manifest)


if __name__ == "__main__":
    main()
