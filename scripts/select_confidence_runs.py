#!/usr/bin/env python3
"""Select seed-42 hyperparameters independently for MSE and Spearman."""

from __future__ import annotations

import argparse

from src.confidence_training.cli import print_json
from src.confidence_training.cli_config import inject_json_config

inject_json_config()
from src.confidence_training.selection import (
    discover_runs,
    select_hyperparameters,
    write_immutable_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON object with CLI-equivalent fields; explicit flags override it")
    parser.add_argument("--runs-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sweep-seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-resolved-config", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved = {
        "runs_roots": args.runs_root,
        "output": args.output,
        "sweep_seed": args.sweep_seed,
    }
    if args.print_resolved_config:
        print_json(resolved)
    selection = select_hyperparameters(
        discover_runs(args.runs_root), sweep_seed=args.sweep_seed
    )
    if args.dry_run:
        print_json(selection)
        return
    write_immutable_json(args.output, selection)
    print_json(selection)


if __name__ == "__main__":
    main()

