"""Argument helpers shared by shell-facing experiment commands."""

from __future__ import annotations

import argparse
import dataclasses
import json
from typing import Any

from .config import ADAPTATION_STRATEGIES, LOSS_TYPES, RunConfig, resolve_run_config


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config")
    parser.add_argument("--run-name")
    parser.add_argument("--dataset-name")
    parser.add_argument("--model-name")
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--adaptation-strategy", choices=ADAPTATION_STRATEGIES)
    parser.add_argument("--loss", choices=LOSS_TYPES)
    parser.add_argument("--train-dir")
    parser.add_argument("--validation-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--probing-dataset-name")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--standardize", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--device")
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--chat-template-kwargs", type=json.loads)
    parser.add_argument("--train-chat-template-kwargs", type=json.loads)
    parser.add_argument("--validation-chat-template-kwargs", type=json.loads)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--log-every-steps",
        type=int,
        help="Log interval training statistics at this optimizer-step cadence (adapted LMs only).",
    )
    parser.add_argument(
        "--save-resume-state", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-resolved-config", action="store_true")


def config_from_namespace(args: argparse.Namespace) -> RunConfig:
    mapping = vars(args)
    excluded = {"config", "dry_run", "print_resolved_config", "stop_after_epoch", "rung"}
    overrides = {
        key.replace("-", "_"): value
        for key, value in mapping.items()
        if key not in excluded and value is not None
    }
    config = resolve_run_config(args.config, overrides)
    if (
        config.adaptation_strategy != "linear_probe"
        and config.log_every_steps is None
    ):
        config = dataclasses.replace(config, log_every_steps=10)
        config.validate()
    return config


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))
