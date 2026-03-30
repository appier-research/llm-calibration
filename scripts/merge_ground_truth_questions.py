#!/usr/bin/env python
"""
Merge the "question" field from a source ground_truth.jsonl into a target file that lacks it.

Rows are matched by example_id. All other fields on each target line are preserved.

Usage:
    uv run python scripts/merge_ground_truth_questions.py \\
        --source outputs/triviaqa-train__Qwen3-8B-non-thinking/ground_truth.jsonl \\
        --target outputs/triviaqa-train__gpt-oss-20b/ground_truth.jsonl \\
        --output outputs/triviaqa-train__gpt-oss-20b/ground_truth.jsonl \\
        --backup

    # Or write to a new file without touching the original:
    uv run python scripts/merge_ground_truth_questions.py \\
        --source ... --target ... --output outputs/triviaqa-train__gpt-oss-20b/ground_truth.with_questions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


def load_questions_by_id(source_path: Path) -> dict[str, str]:
    """example_id -> question"""
    m: dict[str, str] = {}
    with open(source_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            eid = d["example_id"]
            if "question" not in d or d["question"] is None:
                raise ValueError(f"{source_path}:{line_num}: missing question for example_id={eid!r}")
            m[eid] = d["question"]
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge question field from source ground_truth.jsonl into target by example_id")
    parser.add_argument("--source", type=Path, required=True, help="JSONL with example_id and question")
    parser.add_argument("--target", type=Path, required=True, help="JSONL to enrich (missing question)")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write merged JSONL (can equal --target to replace in place)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="If --output equals --target, copy target to target.with_backup_suffix first",
    )
    parser.add_argument(
        "--backup-suffix",
        type=str,
        default=".bak_before_merge_questions",
        help="Suffix for backup file when --backup is set (default: .bak_before_merge_questions)",
    )
    args = parser.parse_args()

    questions = load_questions_by_id(args.source)
    print(f"Loaded {len(questions)} example_id -> question mappings from {args.source}")

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.backup and out_path.resolve() == args.target.resolve():
        bak = args.target.with_name(args.target.name + args.backup_suffix)
        shutil.copy2(args.target, bak)
        print(f"Backed up {args.target} -> {bak}")

    # Write to a temp file first so --output can equal --target (same path) without truncating before read.
    n = 0
    missing: list[str] = []
    fd, tmp_name = tempfile.mkstemp(
        suffix=".jsonl",
        prefix="merge_ground_truth_",
        dir=out_path.parent,
    )
    try:
        os.close(fd)
        tmp_path = Path(tmp_name)
        with open(args.target) as fin, open(tmp_path, "w") as fout:
            for line_num, line in enumerate(fin, 1):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                eid = d["example_id"]
                if eid not in questions:
                    missing.append(eid)
                    continue
                d["question"] = questions[eid]
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
                n += 1

        if missing:
            tmp_path.unlink(missing_ok=True)
            raise SystemExit(
                f"ERROR: {len(missing)} example_ids in target not found in source. First few: {missing[:5]}"
            )

        tmp_path.replace(out_path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise

    print(f"Wrote {n} lines to {out_path}")


if __name__ == "__main__":
    main()
