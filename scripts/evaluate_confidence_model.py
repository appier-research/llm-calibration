#!/usr/bin/env python3
"""Evaluate a selected confidence checkpoint with raw and isotonic predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.confidence_training.artifacts import atomic_write_json
from src.confidence_training.cli import print_json
from src.confidence_training.cli_config import inject_json_config

inject_json_config()
from src.confidence_training.collate import PromptCollator, collate_linear
from src.confidence_training.config import run_config_from_mapping
from src.confidence_training.data import (
    LinearFeatureDataset,
    PromptConfidenceDataset,
    find_split_overlaps,
    load_confidence_examples,
    load_ground_truth,
)
from src.confidence_training.isotonic import apply_isotonic, load_isotonic, metrics
from src.confidence_training.models import load_score_model
from src.confidence_training.training import validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON object with CLI-equivalent fields; explicit flags override it")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--registry")
    parser.add_argument("--model-name")
    parser.add_argument("--dataset-name")
    parser.add_argument("--adaptation-strategy")
    parser.add_argument("--loss")
    parser.add_argument("--selector", choices=("mse", "spearman_r"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-resolved-config", action="store_true")
    return parser.parse_args()


def checkpoint_from_registry(args: argparse.Namespace) -> Path:
    registry = json.loads(Path(args.registry).read_text())
    wanted = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "adaptation_strategy": args.adaptation_strategy,
        "loss": args.loss,
        "selector": args.selector,
    }
    if any(value is None for value in wanted.values()):
        raise ValueError("Registry lookup requires model, dataset, adaptation, loss, and selector")
    matches = [entry for entry in registry["entries"] if entry["condition"] == wanted]
    if len(matches) != 1:
        raise ValueError(f"Expected one registry entry for {wanted}; found {len(matches)}")
    seed_matches = [entry for entry in matches[0]["seeds"] if int(entry["seed"]) == args.seed]
    if len(seed_matches) != 1:
        raise ValueError(f"Registry has no unique seed {args.seed} for {wanted}")
    return Path(seed_matches[0]["checkpoint_path"])


def main() -> None:
    args = parse_args()
    if bool(args.checkpoint_dir) == bool(args.registry):
        raise ValueError("Specify exactly one of --checkpoint-dir or --registry")
    checkpoint = Path(args.checkpoint_dir) if args.checkpoint_dir else checkpoint_from_registry(args)
    run_dir = checkpoint.parent.parent
    config_mapping = json.loads((run_dir / "resolved_config.json").read_text())
    config_mapping.pop("_provenance", None)
    config_mapping["resume"] = False
    config = run_config_from_mapping(config_mapping)
    resolved = {
        "checkpoint_dir": str(checkpoint),
        "test_dir": args.test_dir,
        "output_dir": args.output_dir,
        "device": args.device,
        "batch_size": args.batch_size,
    }
    if args.print_resolved_config or args.dry_run:
        print_json(resolved)
    if args.dry_run:
        return
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    train_gt = load_ground_truth(Path(config.train_dir) / "ground_truth.jsonl")
    validation_gt = load_ground_truth(Path(config.validation_dir) / "ground_truth.jsonl")
    test_gt = load_ground_truth(Path(args.test_dir) / "ground_truth.jsonl")
    for role, rows in (("train", train_gt), ("validation", validation_gt)):
        overlap = find_split_overlaps(rows, test_gt)
        if overlap["example_ids"] or overlap["questions"]:
            raise ValueError(f"Test split overlaps {role}: {overlap}")

    device = torch.device(
        args.device
        if torch.cuda.is_available() or not args.device.startswith("cuda")
        else "cpu"
    )
    model, tokenizer = load_score_model(checkpoint / "model", config, device=device)
    model.eval()
    examples = load_confidence_examples(args.test_dir)
    if config.adaptation_strategy == "linear_probe":
        dataset = LinearFeatureDataset.load(args.test_dir, config.probing_dataset_name)
        if dataset.example_ids != [example.example_id for example in examples]:
            raise ValueError("Test feature order does not match ground truth")
        collate = collate_linear
    else:
        dataset = PromptConfidenceDataset(examples)
        collate = PromptCollator(
            tokenizer,
            device=device,
            max_sequence_length=config.max_sequence_length,
            chat_template_kwargs=config.chat_template_kwargs,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate,
    )
    predictions = validate(model, loader, device)
    isotonic = load_isotonic(checkpoint)
    calibrated = apply_isotonic(predictions.scores.tolist(), isotonic)
    raw_metrics = metrics(predictions.probabilities.tolist(), predictions.targets.tolist())
    calibrated_metrics = metrics(calibrated, predictions.targets.tolist())
    result = {
        "evaluation_version": 1,
        "checkpoint_path": str(checkpoint),
        "raw": raw_metrics,
        "calibrated": calibrated_metrics,
        "n_examples": len(examples),
    }
    atomic_write_json(output / "metrics.json", result)
    ground_truth_by_id = {row.example_id: row for row in test_gt}
    with (output / "predictions.jsonl").open("w") as handle:
        for example_id, score, raw, calibrated_value in zip(
            predictions.example_ids,
            predictions.scores.tolist(),
            predictions.probabilities.tolist(),
            calibrated.tolist(),
            strict=True,
        ):
            truth = ground_truth_by_id[example_id]
            handle.write(
                json.dumps(
                    {
                        "example_id": example_id,
                        "score": float(score),
                        "raw_confidence": float(raw),
                        "isotonic_confidence": float(calibrated_value),
                        "num_samples": truth.num_samples,
                        "num_correct": truth.num_correct,
                        "expected_accuracy": truth.expected_accuracy,
                    }
                )
                + "\n"
            )
    print_json(result)


if __name__ == "__main__":
    main()

