"""Dual-policy successive-halving promotion for confidence sweeps."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import atomic_write_json
from .selection import CONDITION_FIELDS


def _finite(value: Any, *, worst: float) -> float:
    number = float(value)
    return number if math.isfinite(number) else worst


def _history_selector(
    run: dict[str, Any], selector: str, cutoff_epoch: int
) -> dict[str, Any]:
    history_path = Path(run["run_dir"]) / "history.jsonl"
    if not history_path.exists():
        state = run.get("selectors", {}).get(selector, {})
        if state.get("epoch") and int(state["epoch"]) <= cutoff_epoch:
            return {"epoch": int(state["epoch"]), "metrics": state["metrics"]}
        raise ValueError(f"Missing history needed for prefix promotion: {history_path}")
    winner: dict[str, Any] | None = None
    maximum_epoch = 0
    for line in history_path.read_text().splitlines():
        row = json.loads(line)
        epoch = int(row.get("epoch", 0))
        if row.get("kind") == "epoch":
            maximum_epoch = max(maximum_epoch, epoch)
        if row.get("kind") != "epoch" or epoch <= 0 or epoch > cutoff_epoch:
            continue
        metrics = {
            key.removeprefix("validation_"): value
            for key, value in row.items()
            if key.startswith("validation_")
        }
        if not {"mse", "spearman_r"}.issubset(metrics):
            raise ValueError(f"History row lacks promotion metrics: {history_path}:{epoch}")
        candidate = {"epoch": epoch, "metrics": metrics}
        if winner is None:
            winner = candidate
        elif selector == "mse" and float(metrics["mse"]) < float(winner["metrics"]["mse"]):
            winner = candidate
        elif selector == "spearman_r" and float(metrics["spearman_r"]) > float(
            winner["metrics"]["spearman_r"]
        ):
            winner = candidate
    if maximum_epoch < cutoff_epoch:
        raise ValueError(
            f"Run reaches epoch {maximum_epoch}, below promotion cutoff {cutoff_epoch}: {history_path}"
        )
    if winner is None:
        raise ValueError(f"No epoch through cutoff {cutoff_epoch}: {history_path}")
    return winner


def _candidate_key(candidate: dict[str, Any], selector: str) -> tuple[Any, ...]:
    config = candidate["run"]["config"]
    metrics = candidate["selector_state"]["metrics"]
    stable = (
        float(config["learning_rate"]),
        int(config["batch_size"]),
        str(config["run_name"]),
        str(candidate["run"]["run_dir"]),
    )
    if selector == "mse":
        return (
            _finite(metrics["mse"], worst=math.inf),
            -_finite(metrics["spearman_r"], worst=-math.inf),
            *stable,
        )
    return (
        -_finite(metrics["spearman_r"], worst=-math.inf),
        _finite(metrics["mse"], worst=math.inf),
        *stable,
    )


def promote_runs(
    runs: Sequence[dict[str, Any]],
    *,
    group_fields: Sequence[str] = (*CONDITION_FIELDS, "batch_size"),
    cutoff_epoch: int | None = None,
    cutoff_by_run: Mapping[str, int] | None = None,
    cutoff_by_batch: Mapping[str, int] | None = None,
    rung: int = 0,
) -> dict[str, Any]:
    """Promote the union of the MSE and Spearman winners in every group."""

    cutoff_by_run = cutoff_by_run or {}
    cutoff_by_batch = cutoff_by_batch or {}
    eligible = [run for run in runs if run.get("status") in {"paused", "complete"}]
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in eligible:
        config = run["config"]
        group = {field: config[field] for field in group_fields}
        groups.setdefault(json.dumps(group, sort_keys=True), []).append(run)

    entries: list[dict[str, Any]] = []
    promoted: set[str] = set()
    considered: set[str] = set()
    for group_key in sorted(groups):
        group_runs = groups[group_key]
        selected_by: dict[str, dict[str, Any]] = {}
        for selector in ("mse", "spearman_r"):
            candidates: list[dict[str, Any]] = []
            for run in group_runs:
                run_dir = str(Path(run["run_dir"]).resolve())
                run_cutoff = cutoff_by_run.get(run_dir)
                if run_cutoff is None:
                    run_cutoff = cutoff_by_batch.get(str(run["config"]["batch_size"]), cutoff_epoch)
                if run_cutoff is None:
                    run_cutoff = int(
                        run.get("trained_through_epoch", run["config"]["epochs"])
                    )
                state = _history_selector(run, selector, int(run_cutoff))
                candidates.append(
                    {
                        "run": run,
                        "selector_state": state,
                        "cutoff_epoch": int(run_cutoff),
                    }
                )
                considered.add(run_dir)
            winner = min(candidates, key=lambda item: _candidate_key(item, selector))
            run_dir = str(Path(winner["run"]["run_dir"]).resolve())
            promoted.add(run_dir)
            selected_by[selector] = {
                "run_dir": run_dir,
                "cutoff_epoch": winner["cutoff_epoch"],
                "epoch": winner["selector_state"]["epoch"],
                "validation_metrics": winner["selector_state"]["metrics"],
            }
        entries.append(
            {
                "group": json.loads(group_key),
                "candidate_count": len(group_runs),
                "selected_by": selected_by,
                "promoted_run_dirs": sorted(
                    {value["run_dir"] for value in selected_by.values()}
                ),
            }
        )
    return {
        "promotion_version": 1,
        "rung": rung,
        "group_fields": list(group_fields),
        "entries": entries,
        "promoted_run_dirs": sorted(promoted),
        "screened_out_run_dirs": sorted(considered - promoted),
    }


def apply_screening(
    promotion: dict[str, Any], *, promotion_path: str | Path | None = None
) -> list[str]:
    """Mark non-promoted paused/complete manifests as screened out."""

    changed: list[str] = []
    for run_dir in promotion.get("screened_out_run_dirs", []):
        manifest_path = Path(run_dir) / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") not in {"paused", "complete", "screened_out"}:
            raise ValueError(
                f"Cannot screen run with status={manifest.get('status')}: {run_dir}"
            )
        if manifest.get("status") == "screened_out":
            continue
        manifest["status"] = "screened_out"
        manifest["screened_out_reason"] = f"not promoted after rung {promotion['rung']}"
        if promotion_path is not None:
            manifest["promotion_manifest"] = str(promotion_path)
        atomic_write_json(manifest_path, manifest)
        changed.append(run_dir)
    return changed
