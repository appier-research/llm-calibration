"""Materialize shell-friendly linear-probe sweep and continuation jobs."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import RunConfig, run_config_from_mapping
from .wandb_logging import linear_probe_project


BATCH_SIZES = (8, 16, 32)


def default_linear_probe_learning_rates(
    model_name: str, dataset_name: str, loss: str
) -> tuple[float, ...]:
    model = model_name.casefold()
    dataset = dataset_name.casefold()
    if "olmo" in model:
        return (5e-4, 1e-3, 2e-3, 5e-3, 1e-2)
    if "qwen" in model:
        if dataset == "math":
            return (1e-5, 2e-5, 5e-5, 1e-4, 2e-4)
        return (2e-5, 5e-5, 1e-4, 2e-4, 5e-4)
    if "gpt-oss" in model:
        if dataset == "math" and loss == "bce":
            return (5e-4, 1e-3, 2e-3, 5e-3, 1e-2)
        if dataset in {"math", "mmlu"}:
            return (2e-4, 5e-4, 1e-3, 2e-3, 5e-3)
        return (2e-5, 5e-5, 1e-4, 2e-4, 5e-4)
    raise ValueError(f"No default learning-rate grid for model={model_name!r}")


def _condition(run: dict[str, Any]) -> tuple[str, str, str]:
    config = run["config"]
    return config["model_name"], config["dataset_name"], config["loss"]


def _epoch_seconds(run_dir: str | Path) -> float:
    rows = [
        json.loads(line)
        for line in (Path(run_dir) / "history.jsonl").read_text().splitlines()
        if line.strip()
    ]
    timings = [float(row["elapsed_seconds"]) for row in rows if "elapsed_seconds" in row]
    if not timings:
        raise ValueError(f"Pilot lacks epoch timings: {run_dir}")
    return statistics.median(timings)


def _lookup(mapping: Mapping[str, Any], batch_size: int, name: str) -> int:
    try:
        value = mapping[str(batch_size)]
    except KeyError as error:
        raise ValueError(f"{name} must define batch size {batch_size}") from error
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name}[{batch_size}] must be positive")
    return value


def build_initial_linear_probe_plan(
    pilot_runs: Sequence[dict[str, Any]],
    *,
    output_root: str | Path,
    config_root: str | Path,
    max_epochs_by_batch: Mapping[str, Any],
    stop_epochs_by_batch: Mapping[str, Any],
    learning_rate_overrides: Mapping[str, Sequence[float]] | None = None,
    seed: int = 42,
    wandb_mode: str = "online",
    timing_safety_factor: float = 1.25,
    reuse_complete_pilots: bool = True,
) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    """Build 27 x 3 x 5 rung-zero jobs, optionally reusing exact pilots."""

    if timing_safety_factor <= 0:
        raise ValueError("timing_safety_factor must be positive")
    bases: dict[tuple[str, str, str], dict[str, Any]] = {}
    reusable: set[tuple[str, str, str, float, int, int]] = set()
    for run in pilot_runs:
        config = run["config"]
        if config.get("adaptation_strategy") != "linear_probe":
            continue
        condition = _condition(run)
        bases.setdefault(condition, run)
        if run.get("status") == "complete":
            reusable.add(
                (*condition, float(config["learning_rate"]), int(config["batch_size"]), int(config["seed"]))
            )
    if len(bases) != 27:
        raise ValueError(f"Expected 27 linear-probe pilot arms; found {len(bases)}")

    output_root = Path(output_root)
    config_root = Path(config_root)
    jobs: list[dict[str, Any]] = []
    configs: list[tuple[Path, dict[str, Any]]] = []
    reused: list[dict[str, Any]] = []
    overrides = learning_rate_overrides or {}
    for condition in sorted(bases):
        model_name, dataset_name, loss = condition
        pilot = bases[condition]
        pilot_config = run_config_from_mapping(pilot["config"])
        override_key = "__".join(condition)
        rates = tuple(
            float(value)
            for value in overrides.get(
                override_key,
                default_linear_probe_learning_rates(model_name, dataset_name, loss),
            )
        )
        if len(rates) != 5 or len(set(rates)) != 5:
            raise ValueError(f"Every arm requires five distinct learning rates: {condition}")
        pilot_epoch_seconds = _epoch_seconds(pilot["run_dir"])
        for batch_size in BATCH_SIZES:
            max_epochs = _lookup(max_epochs_by_batch, batch_size, "max_epochs_by_batch")
            stop_epoch = _lookup(stop_epochs_by_batch, batch_size, "stop_epochs_by_batch")
            if stop_epoch > max_epochs:
                raise ValueError("Rung stop epoch cannot exceed maximum epochs")
            for learning_rate in rates:
                reuse_key = (*condition, learning_rate, batch_size, seed)
                if reuse_complete_pilots and reuse_key in reusable:
                    reused.append(
                        {
                            "model_name": model_name,
                            "dataset_name": dataset_name,
                            "loss": loss,
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                            "seed": seed,
                        }
                    )
                    continue
                run_name = (
                    f"{dataset_name}__{model_name}__linear_probe__{loss}__"
                    f"lr-{learning_rate:g}__bs-{batch_size}__seed-{seed}"
                )
                values = pilot_config.to_dict()
                values.update(
                    {
                        "run_name": run_name,
                        "output_dir": str(output_root / run_name),
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "epochs": max_epochs,
                        "seed": seed,
                        "resume": False,
                        "save_resume_state": False,
                        "wandb_project": linear_probe_project(model_name, dataset_name),
                        "wandb_mode": wandb_mode,
                    }
                )
                resolved = run_config_from_mapping(values).to_dict()
                config_path = config_root / f"{run_name}.json"
                configs.append((config_path, resolved))
                predicted = (
                    pilot_epoch_seconds
                    * stop_epoch
                    * (int(pilot_config.batch_size) / batch_size)
                    * timing_safety_factor
                )
                jobs.append(
                    {
                        "job_id": run_name,
                        "command": [
                            "uv",
                            "run",
                            "python",
                            "scripts/train_confidence_model.py",
                            "--config",
                            str(config_path),
                            "--stop-after-epoch",
                            str(stop_epoch),
                            "--rung",
                            "0",
                        ],
                        "predicted_seconds": predicted,
                        "condition": {
                            "model_name": model_name,
                            "dataset_name": dataset_name,
                            "loss": loss,
                            "batch_size": batch_size,
                            "learning_rate": learning_rate,
                        },
                    }
                )
    return (
        {
            "plan_version": 1,
            "kind": "linear_probe_rung_0",
            "job_count": len(jobs),
            "reused_pilot_count": len(reused),
            "reused_pilots": reused,
            "jobs": jobs,
        },
        configs,
    )


def build_continuation_plan(
    promotion: dict[str, Any],
    runs: Sequence[dict[str, Any]],
    *,
    stop_epochs_by_batch: Mapping[str, Any],
    rung: int,
    timing_safety_factor: float = 1.25,
) -> dict[str, Any]:
    promoted = {str(Path(path).resolve()) for path in promotion["promoted_run_dirs"]}
    by_path = {str(Path(run["run_dir"]).resolve()): run for run in runs}
    jobs: list[dict[str, Any]] = []
    already_complete: list[str] = []
    for run_dir in sorted(promoted):
        run = by_path.get(run_dir)
        if run is None:
            raise ValueError(f"Promoted run was not discovered: {run_dir}")
        if run.get("status") == "complete":
            already_complete.append(run_dir)
            continue
        if run.get("status") != "paused":
            raise ValueError(f"Promoted run is not resumable: {run_dir}")
        config = run["config"]
        target = _lookup(stop_epochs_by_batch, int(config["batch_size"]), "stop_epochs_by_batch")
        current = int(run.get("trained_through_epoch", 0))
        if target <= current or target > int(config["epochs"]):
            raise ValueError(f"Invalid continuation target {target} for {run_dir} at {current}")
        predicted = _epoch_seconds(run_dir) * (target - current) * timing_safety_factor
        jobs.append(
            {
                "job_id": f"{config['run_name']}__rung-{rung}",
                "command": [
                    "uv",
                    "run",
                    "python",
                    "scripts/train_confidence_model.py",
                    "--config",
                    str(Path(run_dir) / "resolved_config.json"),
                    "--resume",
                    "--stop-after-epoch",
                    str(target),
                    "--rung",
                    str(rung),
                ],
                "predicted_seconds": predicted,
                "run_dir": run_dir,
            }
        )
    return {
        "plan_version": 1,
        "kind": f"confidence_continuation_rung_{rung}",
        "job_count": len(jobs),
        "already_complete": already_complete,
        "jobs": jobs,
    }
