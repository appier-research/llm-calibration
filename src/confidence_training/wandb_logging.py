"""Resume-safe W&B logging for local-authoritative confidence runs."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Iterable

from .config import RunConfig


def linear_probe_project(model_name: str, dataset_name: str) -> str:
    """Return the prescribed one-project-per-model/dataset name."""

    return f"linear-probe__{model_name}__{dataset_name}"


def full_weight_project(model_name: str, dataset_name: str) -> str:
    return f"lm-full-weight__{model_name}__{dataset_name}"


def deterministic_wandb_id(config: RunConfig) -> str:
    payload = {
        "scientific_config_hash": config.scientific_hash(),
        "run_name": config.run_name,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:32]


def wandb_run_name(config: RunConfig) -> str:
    return (
        f"loss-{config.loss}__bs-{config.batch_size}__"
        f"lr-{config.learning_rate:g}__seed-{config.seed}"
    )


def normalize_history_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "epoch": int(row["epoch"]),
        "optimizer_steps": int(row.get("optimizer_steps", 0)),
        "rung": int(row.get("rung", 0)),
    }
    if row.get("kind") == "validation":
        for key, value in row.items():
            if key not in {"kind", "epoch", "optimizer_steps", "rung"}:
                normalized[f"validation/{key}"] = value
        return normalized
    if row.get("kind") == "train_step":
        mappings = {
            "epoch_step": "train/epoch_step",
            "examples_seen_in_epoch": "train/examples_seen_in_epoch",
            "interval_examples": "train/interval_examples",
            "interval_train_loss": "train/loss",
            "interval_gradient_norm_mean": "train/gradient_norm",
            "interval_seconds": "train/interval_seconds",
            "learning_rate": "train/learning_rate",
        }
        for source, destination in mappings.items():
            if source in row:
                normalized[destination] = row[source]
        return normalized
    mappings = {
        "train_loss": "train/epoch_loss",
        "gradient_norm_mean": "train/epoch_gradient_norm",
        "elapsed_seconds": "train/epoch_seconds",
    }
    for source, destination in mappings.items():
        if source in row:
            normalized[destination] = row[source]
    for key, value in row.items():
        if key.startswith("validation_"):
            normalized[f"validation/{key.removeprefix('validation_')}"] = value
    return normalized


class WandbLogger:
    def __init__(self, run: Any) -> None:
        self.run = run
        self.failed = False

    @classmethod
    def create(
        cls,
        config: RunConfig,
        resolved: dict[str, Any],
        *,
        wandb_module: Any | None = None,
        project: str | None = None,
        entity: str | None = None,
        mode: str | None = None,
    ) -> "WandbLogger | None":
        selected_mode = mode or config.wandb_mode
        if selected_mode == "disabled":
            return None
        if wandb_module is None:
            import wandb as wandb_module

        selected_project = project or config.wandb_project
        if selected_project is None and config.adaptation_strategy == "linear_probe":
            selected_project = linear_probe_project(config.model_name, config.dataset_name)
        if selected_project is None and config.adaptation_strategy == "full_weight":
            selected_project = full_weight_project(config.model_name, config.dataset_name)
        if selected_project is None:
            raise ValueError("wandb_project is required for non-linear adaptation strategies")
        try:
            run = wandb_module.init(
                project=selected_project,
                entity=entity or config.wandb_entity,
                id=deterministic_wandb_id(config),
                resume="allow",
                name=wandb_run_name(config),
                group=f"loss-{config.loss}",
                tags=[
                    "confidence-training",
                    config.adaptation_strategy,
                    config.loss,
                    config.model_name,
                    config.dataset_name,
                ],
                mode=selected_mode,
                config=resolved,
            )
        except Exception as error:  # local artifacts remain authoritative
            warnings.warn(f"W&B initialization failed; continue locally: {error}")
            return None
        if hasattr(run, "define_metric"):
            run.define_metric("optimizer_steps")
            run.define_metric("*", step_metric="optimizer_steps")
        return cls(run)

    def log(self, row: dict[str, Any]) -> None:
        if self.failed:
            return
        normalized = normalize_history_row(row)
        try:
            self.run.log(normalized)
        except Exception as error:  # preserve scientific training on network failure
            self.failed = True
            warnings.warn(f"W&B logging failed; continue locally for backfill: {error}")

    def update_summary(self, manifest: dict[str, Any]) -> None:
        if self.failed:
            return
        selectors = manifest.get("selectors", {})
        values = {
            "status": manifest.get("status"),
            "trained_through_epoch": manifest.get("trained_through_epoch"),
            "rung": manifest.get("rung", 0),
            **{
                f"runtime/{key}": value
                for key, value in manifest.get("runtime_summary", {}).items()
            },
        }
        report = manifest.get("parameter_report") or manifest.get("provenance", {}).get(
            "parameter_report", {}
        )
        for key, value in report.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                values[f"parameters/{key}"] = value
        for selector in ("mse", "spearman_r"):
            state = selectors.get(selector, {})
            values[f"best/{selector}"] = state.get("value")
            values[f"best/{selector}_epoch"] = state.get("epoch")
            values[f"artifacts/{selector}_checkpoint"] = state.get("checkpoint_path")
            checkpoint_path = state.get("checkpoint_path")
            if checkpoint_path:
                checkpoint_manifest = Path(checkpoint_path) / "checkpoint_manifest.json"
                if checkpoint_manifest.is_file():
                    values[f"artifacts/{selector}_manifest"] = str(checkpoint_manifest)
                    values[f"artifacts/{selector}_model_sha256"] = json.loads(
                        checkpoint_manifest.read_text()
                    ).get("model_sha256")
        try:
            if hasattr(self.run.summary, "update"):
                self.run.summary.update(values)
            else:
                for key, value in values.items():
                    self.run.summary[key] = value
        except Exception as error:
            self.failed = True
            warnings.warn(f"W&B summary failed; continue locally for backfill: {error}")

    def finish(self) -> None:
        try:
            self.run.finish()
        except Exception as error:
            warnings.warn(f"W&B finish failed; local artifacts are complete: {error}")


def read_history(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def backfill_history(logger: WandbLogger, rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        logger.log(row)
