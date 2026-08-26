#!/usr/bin/env python3
"""Audit confidence datasets without modifying or reconstructing them."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.confidence_training.artifacts import atomic_write_json
from src.confidence_training.cli import print_json
from src.confidence_training.data import (
    AUDIT_SCOPES,
    audit_dataset_folder,
    find_split_overlaps,
    load_ground_truth,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON with an entries list")
    parser.add_argument(
        "--entry",
        action="append",
        nargs=4,
        metavar=("DATASET", "ROLE", "MODEL", "FOLDER"),
        default=[],
    )
    parser.add_argument("--probing-dataset-name", default="probing_dataset_cc")
    parser.add_argument("--scope", choices=AUDIT_SCOPES)
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-resolved-config", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries: list[dict[str, str]] = []
    configured_scope = None
    if args.config:
        value = json.loads(Path(args.config).read_text())
        entries.extend(value["entries"])
        configured_scope = value.get("audit_scope")
    entries.extend(
        {"dataset": dataset, "role": role, "model_name": model, "path": path}
        for dataset, role, model, path in args.entry
    )
    if not entries:
        raise ValueError("Provide --config or at least one --entry")
    scope = args.scope or configured_scope or "probing_artifacts"
    if scope not in AUDIT_SCOPES:
        raise ValueError(f"Unknown audit scope: {scope}")
    resolved = {
        "entries": entries,
        "audit_scope": scope,
        "probing_dataset_name": args.probing_dataset_name,
        "output": args.output,
    }
    if args.print_resolved_config or args.dry_run:
        print_json(resolved)
    if args.dry_run:
        return
    reports = []
    by_split: dict[tuple[str, str], list[tuple[dict[str, str], dict[str, Any]]]] = defaultdict(list)
    for entry in entries:
        report = audit_dataset_folder(
            entry["path"],
            scope=scope,
            probing_dataset_name=args.probing_dataset_name,
            expected_count=entry.get("expected_count"),
        )
        reports.append({"entry": entry, **report})
        by_split[(entry["dataset"], entry["role"])].append((entry, report))
    cross_model_errors: list[str] = []
    for key, group in by_split.items():
        valid = [(entry, report) for entry, report in group if report["ok"]]
        if valid:
            reference = valid[0][1]
            for entry, report in valid[1:]:
                if report["example_ids"] != reference["example_ids"]:
                    cross_model_errors.append(
                        f"{key}: ordered ID mismatch for {entry['model_name']}"
                    )
                reference_questions = dict(
                    zip(
                        reference["example_ids"],
                        reference["normalized_questions"],
                        strict=True,
                    )
                )
                questions = dict(
                    zip(
                        report["example_ids"],
                        report["normalized_questions"],
                        strict=True,
                    )
                )
                common_ids = reference_questions.keys() & questions.keys()
                if any(
                    reference_questions[example_id] != questions[example_id]
                    for example_id in common_ids
                ):
                    cross_model_errors.append(
                        f"{key}: canonical-question mismatch by ID for {entry['model_name']}"
                    )
    overlaps: list[dict[str, Any]] = []
    by_dataset_model: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for entry in entries:
        by_dataset_model[(entry["dataset"], entry["model_name"])].append(entry)
    for key, group in by_dataset_model.items():
        for index, first in enumerate(group):
            for second in group[index + 1 :]:
                overlap = find_split_overlaps(
                    load_ground_truth(Path(first["path"]) / "ground_truth.jsonl"),
                    load_ground_truth(Path(second["path"]) / "ground_truth.jsonl"),
                )
                if overlap["example_ids"] or overlap["questions"]:
                    overlaps.append(
                        {"dataset_model": key, "roles": [first["role"], second["role"]], **overlap}
                    )
    public_reports = [
        {
            key: value
            for key, value in report.items()
            if key not in {"example_ids", "questions", "normalized_questions"}
        }
        for report in reports
    ]
    result = {
        "audit_version": 2,
        "audit_scope": scope,
        "ok": all(report["ok"] for report in reports)
        and not cross_model_errors
        and not overlaps,
        "datasets": public_reports,
        "cross_model_errors": cross_model_errors,
        "split_overlaps": overlaps,
    }
    if args.output:
        atomic_write_json(args.output, result)
    print_json(result)
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
