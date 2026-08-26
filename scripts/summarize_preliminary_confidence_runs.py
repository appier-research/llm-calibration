#!/usr/bin/env python3
"""Validate and summarize a directory of completed preliminary confidence runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

from src.confidence_training.artifacts import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--expected-runs", type=int, default=27)
    parser.add_argument("--near-ceiling-fraction", type=float, default=0.9)
    parser.add_argument("--audit-report")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def summarize_run(run_dir: Path, near_ceiling_fraction: float) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    manifest_path = run_dir / "run_manifest.json"
    history_path = run_dir / "history.jsonl"
    manifest = json.loads(manifest_path.read_text())
    config = manifest["config"]
    history = load_jsonl(history_path)
    epoch_rows = [row for row in history if row.get("kind") == "epoch"]
    expected_epochs = int(config["epochs"])
    observed_epochs = [int(row["epoch"]) for row in epoch_rows]
    if manifest.get("status") != "complete":
        errors.append(f"{run_dir.name}: status={manifest.get('status')}")
    if observed_epochs != list(range(1, expected_epochs + 1)):
        errors.append(f"{run_dir.name}: history does not contain epochs 1..{expected_epochs}")
    initial_rows = [row for row in history if row.get("kind") == "validation" and row.get("epoch") == 0]
    if len(initial_rows) != 1:
        errors.append(f"{run_dir.name}: expected one step-zero validation row")

    selectors = manifest.get("selectors", {})
    for policy in ("mse", "spearman_r"):
        selector = selectors.get(policy, {})
        checkpoint = Path(selector.get("checkpoint_path", ""))
        validation = Path(selector.get("validation_path", ""))
        if not checkpoint.is_dir() or not validation.is_file():
            errors.append(f"{run_dir.name}: missing retained artifact for {policy}")
        elif not (checkpoint / "checkpoint_manifest.json").is_file():
            errors.append(f"{run_dir.name}: missing checkpoint manifest for {policy}")
    checkpoint_dirs = list((run_dir / "checkpoints").glob("epoch_*"))
    referenced = {
        Path(selector["checkpoint_path"]).resolve()
        for selector in selectors.values()
        if selector.get("checkpoint_path")
    }
    if len(checkpoint_dirs) > 2 or {path.resolve() for path in checkpoint_dirs} != referenced:
        errors.append(f"{run_dir.name}: retained checkpoints do not exactly match selectors")

    elapsed = [float(row["elapsed_seconds"]) for row in epoch_rows]
    if config.get("adaptation_strategy") == "full_weight":
        fixed = {
            "learning_rate": 1e-5,
            "batch_size": 16,
            "epochs": 3,
            "seed": 42,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "gradient_checkpointing": True,
            "num_workers": 0,
            "log_every_steps": 10,
            "save_resume_state": False,
        }
        for key, expected in fixed.items():
            if config.get(key) != expected:
                errors.append(
                    f"{run_dir.name}: fixed full-weight field {key}={config.get(key)!r}, "
                    f"expected {expected!r}"
                )
        if config.get("model_name") not in {
            "Olmo-3-7B-Instruct",
            "Qwen3-8B-non-thinking",
        }:
            errors.append(f"{run_dir.name}: unsupported full-weight model")
    mse_selector = selectors["mse"]
    spearman_selector = selectors["spearman_r"]
    max_winner_epoch = max(int(mse_selector["epoch"]), int(spearman_selector["epoch"]))
    row = {
        "run_name": config["run_name"],
        "model_name": config["model_name"],
        "dataset_name": config["dataset_name"],
        "loss": config["loss"],
        "adaptation_strategy": config["adaptation_strategy"],
        "learning_rate": float(config["learning_rate"]),
        "batch_size": int(config["batch_size"]),
        "seed": int(config["seed"]),
        "epochs": expected_epochs,
        "mse_epoch": int(mse_selector["epoch"]),
        "mse": float(mse_selector["metrics"]["mse"]),
        "spearman_at_mse": float(mse_selector["metrics"]["spearman_r"]),
        "spearman_epoch": int(spearman_selector["epoch"]),
        "spearman_r": float(spearman_selector["metrics"]["spearman_r"]),
        "mse_at_spearman": float(spearman_selector["metrics"]["mse"]),
        "max_winner_epoch": max_winner_epoch,
        "winner_near_ceiling": max_winner_epoch >= math.ceil(near_ceiling_fraction * expected_epochs),
        "total_epoch_seconds": sum(elapsed),
        "per_epoch_seconds": json.dumps(elapsed),
        "total_wall_seconds": float(
            manifest.get("runtime_summary", {}).get("wall_seconds", sum(elapsed))
        ),
        "mean_epoch_seconds": statistics.fmean(elapsed),
        "max_epoch_seconds": max(elapsed),
        "checkpoint_count": len(checkpoint_dirs),
        "run_dir": str(run_dir),
    }
    return row, errors


def main() -> None:
    args = parse_args()
    if not 0.0 < args.near_ceiling_fraction <= 1.0:
        raise ValueError("--near-ceiling-fraction must be in (0, 1]")
    root = Path(args.root)
    manifest_paths = sorted(root.glob("*/run_manifest.json"))
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for manifest_path in manifest_paths:
        row, run_errors = summarize_run(manifest_path.parent, args.near_ceiling_fraction)
        rows.append(row)
        errors.extend(run_errors)
    rows.sort(key=lambda row: (row["dataset_name"], row["model_name"], row["loss"]))
    if len(rows) != args.expected_runs:
        errors.append(f"expected {args.expected_runs} runs, found {len(rows)}")
    logical_keys = [
        (row["model_name"], row["dataset_name"], row["loss"], row["adaptation_strategy"])
        for row in rows
    ]
    if len(logical_keys) != len(set(logical_keys)):
        errors.append("duplicate logical model/dataset/loss/adaptation runs found")

    audit_summary = None
    if args.audit_report:
        audit = json.loads(Path(args.audit_report).read_text())
        audit_summary = {
            "path": args.audit_report,
            "ok": audit["ok"],
            "folder_failures": [
                {"entry": item["entry"], "errors": item["errors"]}
                for item in audit["datasets"]
                if not item["ok"]
            ],
            "cross_model_errors": audit["cross_model_errors"],
            "split_overlaps": [
                {
                    "dataset_model": item["dataset_model"],
                    "roles": item["roles"],
                    "example_id_count": len(item["example_ids"]),
                    "question_count": len(item["questions"]),
                }
                for item in audit["split_overlaps"]
            ],
        }

    summary = {
        "summary_version": 1,
        "root": str(root),
        "expected_runs": args.expected_runs,
        "completed_runs": len(rows),
        "integrity_ok": not errors,
        "integrity_errors": errors,
        "near_ceiling_fraction": args.near_ceiling_fraction,
        "near_ceiling_runs": [row["run_name"] for row in rows if row["winner_near_ceiling"]],
        "total_epoch_process_seconds": sum(row["total_epoch_seconds"] for row in rows),
        "mean_epoch_seconds_by_loss": {
            loss: statistics.fmean(
                row["mean_epoch_seconds"] for row in rows if row["loss"] == loss
            )
            for loss in ("mse", "bce", "ranknet")
        },
        "audit": audit_summary,
        "runs": rows,
    }
    print(
        json.dumps(
            {
                "completed_runs": len(rows),
                "integrity_ok": not errors,
                "near_ceiling_runs": len(summary["near_ceiling_runs"]),
                "total_epoch_process_seconds": summary["total_epoch_process_seconds"],
                "mean_epoch_seconds_by_loss": summary["mean_epoch_seconds_by_loss"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if errors:
        raise SystemExit(1)
    atomic_write_json(args.output_json, summary)
    atomic_write_csv(Path(args.output_csv), rows)


if __name__ == "__main__":
    main()
