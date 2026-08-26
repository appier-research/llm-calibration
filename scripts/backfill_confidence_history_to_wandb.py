#!/usr/bin/env python3
"""Backfill local confidence-training histories to deterministic W&B runs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.confidence_training.artifacts import atomic_write_json
from src.confidence_training.cli import print_json
from src.confidence_training.config import run_config_from_mapping
from src.confidence_training.selection import discover_runs
from src.confidence_training.wandb_logging import (
    WandbLogger,
    backfill_history,
    deterministic_wandb_id,
    linear_probe_project,
    read_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", action="append", required=True)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument(
        "--adaptation-strategy",
        action="append",
        choices=("linear_probe", "lora", "full_weight"),
        help="Restrict uploads; repeat to select multiple strategies.",
    )
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continue polling and upload newly completed runs exactly once.",
    )
    parser.add_argument("--expected-runs", type=int)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--state-path",
        help="Durable upload state (required with --watch and multiple roots).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-resolved-config", action="store_true")
    return parser.parse_args()


def _project(config: dict[str, Any]) -> str:
    project = config.get("wandb_project")
    if project:
        return str(project)
    if config["adaptation_strategy"] == "linear_probe":
        return linear_probe_project(config["model_name"], config["dataset_name"])
    raise ValueError(
        f"Run {config.get('run_name', '<unknown>')} has no wandb_project"
    )


def _eligible_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = set(args.adaptation_strategy or ())
    return [
        run
        for run in discover_runs(args.runs_root)
        if (not selected or run["config"]["adaptation_strategy"] in selected)
        and (args.include_incomplete or run.get("status") == "complete")
    ]


def _plan(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "run_dir": run["run_dir"],
            "project": _project(run["config"]),
            "wandb_id": deterministic_wandb_id(
                run_config_from_mapping(run["config"])
            ),
            "status": run.get("status"),
        }
        for run in runs
    ]


def _backfill_run(run: dict[str, Any], args: argparse.Namespace) -> str:
    config = run_config_from_mapping(run["config"])
    logger = WandbLogger.create(
        config,
        config.to_dict(),
        project=_project(run["config"]),
        entity=args.wandb_entity,
        mode=args.wandb_mode,
    )
    assert logger is not None
    try:
        backfill_history(logger, read_history(Path(run["run_dir"]) / "history.jsonl"))
        logger.update_summary(run)
    finally:
        logger.finish()
    return deterministic_wandb_id(config)


def _state_path(args: argparse.Namespace) -> Path:
    if args.state_path:
        return Path(args.state_path)
    if len(args.runs_root) != 1:
        raise ValueError("--state-path is required with --watch and multiple roots")
    return Path(args.runs_root[0]) / "wandb_backfill_state.json"


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "uploaded": {}}
    value = json.loads(path.read_text())
    if value.get("version") != 1 or not isinstance(value.get("uploaded"), dict):
        raise ValueError(f"Invalid W&B backfill state: {path}")
    return value


def _run_once(args: argparse.Namespace, state: dict[str, Any] | None = None) -> int:
    runs = _eligible_runs(args)
    if args.print_resolved_config or args.dry_run:
        print_json(
            {
                "runs_roots": args.runs_root,
                "wandb_entity": args.wandb_entity,
                "wandb_mode": args.wandb_mode,
                "runs": _plan(runs),
            }
        )
    if args.dry_run:
        return 0
    uploaded = state["uploaded"] if state is not None else {}
    count = 0
    for run in runs:
        config = run_config_from_mapping(run["config"])
        run_id = deterministic_wandb_id(config)
        if run_id in uploaded:
            continue
        uploaded_id = _backfill_run(run, args)
        uploaded[uploaded_id] = {
            "run_dir": run["run_dir"],
            "project": _project(run["config"]),
            "uploaded_at_unix": time.time(),
        }
        count += 1
    return count


def main() -> None:
    args = parse_args()
    if args.expected_runs is not None and args.expected_runs <= 0:
        raise ValueError("--expected-runs must be positive")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if not args.watch:
        if args.dry_run:
            _run_once(args)
            return
        state_path = _state_path(args)
        state = _read_state(state_path)
        changed = _run_once(args, state)
        if changed:
            atomic_write_json(state_path, state)
            print_json(
                {
                    "new_uploads": changed,
                    "total_uploads": len(state["uploaded"]),
                    "state_path": str(state_path),
                }
            )
        return

    state_path = _state_path(args)
    state = _read_state(state_path)
    while True:
        changed = _run_once(args, state)
        if changed:
            atomic_write_json(state_path, state)
            print_json(
                {
                    "new_uploads": changed,
                    "total_uploads": len(state["uploaded"]),
                    "state_path": str(state_path),
                }
            )
        if args.expected_runs is not None and len(state["uploaded"]) >= args.expected_runs:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
