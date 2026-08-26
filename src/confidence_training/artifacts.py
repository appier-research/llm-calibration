"""Run-local manifests and rolling dual-selector checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from safetensors.torch import save_file

from .config import RunConfig


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def sha256_path(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    paths = (
        [source]
        if source.is_file()
        else sorted(item for item in source.rglob("*") if item.is_file())
    )
    for item in paths:
        if source.is_dir():
            digest.update(str(item.relative_to(source)).encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class RunArtifacts:
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.root = config.output_path
        self.manifest_path = self.root / "run_manifest.json"
        self.selector_path = self.root / "selectors.json"
        self.history_path = self.root / "history.jsonl"
        self.checkpoint_root = self.root / "checkpoints"

    def initialize(self, provenance: dict[str, Any]) -> None:
        if self.root.exists() and any(self.root.iterdir()):
            if not self.config.resume:
                raise FileExistsError(
                    f"Output directory is nonempty: {self.root}; use resume only for an identical run"
                )
            existing = json.loads((self.root / "resolved_config.json").read_text())
            existing.pop("_provenance", None)
            existing.pop("_runtime", None)
            existing_config = RunConfig(
                **{
                    **existing,
                    "adamw_betas": tuple(existing["adamw_betas"]),
                    "lora_target_modules": tuple(existing["lora_target_modules"]),
                    "lora_target_parameters": tuple(
                        existing.get("lora_target_parameters", [])
                    ),
                }
            )
            if existing_config.scientific_hash() != self.config.scientific_hash():
                raise ValueError("Resume configuration does not match the existing run")
            manifest = json.loads(self.manifest_path.read_text())
            if manifest.get("status") in {"complete", "screened_out"}:
                raise ValueError(
                    f"Cannot resume a run with status={manifest.get('status')!r}"
                )
            manifest["status"] = "running"
            manifest.pop("error", None)
            atomic_write_json(self.manifest_path, manifest)
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_root.mkdir()
        parameter_report = provenance.get("parameter_report")
        runtime = {
            "parameter_report": parameter_report,
            "actual_dtypes": (parameter_report or {}).get("parameters_by_dtype", {}),
        }
        atomic_write_json(
            self.root / "resolved_config.json",
            {**self.config.to_dict(), "_provenance": provenance, "_runtime": runtime},
        )
        selectors = {
            "mse": {"mode": "min", "value": None, "epoch": None},
            "spearman_r": {"mode": "max", "value": None, "epoch": None},
        }
        atomic_write_json(self.selector_path, selectors)
        atomic_write_json(
            self.manifest_path,
            {
                "artifact_version": 1,
                "status": "running",
                "scientific_config_hash": self.config.scientific_hash(),
                "config": self.config.to_dict(),
                "provenance": provenance,
                **runtime,
                "selectors": selectors,
            },
        )

    def append_history(self, row: dict[str, Any]) -> None:
        with self.history_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def load_selectors(self) -> dict[str, dict[str, Any]]:
        return json.loads(self.selector_path.read_text())

    def improvements(self, epoch: int, metrics: dict[str, float]) -> set[str]:
        if epoch == 0:
            return set()
        selectors = self.load_selectors()
        improved: set[str] = set()
        if selectors["mse"]["value"] is None or metrics["mse"] < selectors["mse"]["value"]:
            improved.add("mse")
        if (
            selectors["spearman_r"]["value"] is None
            or metrics["spearman_r"] > selectors["spearman_r"]["value"]
        ):
            improved.add("spearman_r")
        return improved

    def save_improved_checkpoint(
        self,
        *,
        epoch: int,
        metrics: dict[str, float],
        example_ids: Sequence[str],
        scores: torch.Tensor,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        improved: set[str],
        model_saver: Callable[[Path], None],
    ) -> Path | None:
        if not improved:
            return None
        final_path = self.checkpoint_root / f"epoch_{epoch:06d}"
        if not final_path.exists():
            temporary = Path(
                tempfile.mkdtemp(prefix=f".epoch_{epoch:06d}.", dir=self.checkpoint_root)
            )
            try:
                model_saver(temporary / "model")
                save_file(
                    {
                        "scores": scores.detach().cpu().float().contiguous().clone(),
                        "probabilities": probabilities.detach().cpu().float().contiguous().clone(),
                        "targets": targets.detach().cpu().float().contiguous().clone(),
                    },
                    temporary / "validation.safetensors",
                )
                atomic_write_json(temporary / "example_ids.json", list(example_ids))
                atomic_write_json(temporary / "metrics.json", metrics)
                atomic_write_json(
                    temporary / "checkpoint_manifest.json",
                    {
                        "checkpoint_version": 1,
                        "epoch": epoch,
                        "model_sha256": sha256_path(temporary / "model"),
                        "validation_sha256": sha256_path(
                            temporary / "validation.safetensors"
                        ),
                        "example_ids_sha256": sha256_path(temporary / "example_ids.json"),
                        "metrics_sha256": sha256_path(temporary / "metrics.json"),
                    },
                )
                os.replace(temporary, final_path)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)

        selectors = self.load_selectors()
        for selector in improved:
            selectors[selector] = {
                "mode": "min" if selector == "mse" else "max",
                "value": metrics[selector],
                "epoch": epoch,
                "checkpoint_path": str(final_path),
                "validation_path": str(final_path / "validation.safetensors"),
                "metrics": metrics,
            }
        atomic_write_json(self.selector_path, selectors)
        self._purge_unreferenced(selectors)
        manifest = json.loads(self.manifest_path.read_text())
        manifest["selectors"] = selectors
        atomic_write_json(self.manifest_path, manifest)
        return final_path

    def _purge_unreferenced(self, selectors: dict[str, dict[str, Any]]) -> None:
        referenced = {
            Path(value["checkpoint_path"]).resolve()
            for value in selectors.values()
            if value.get("checkpoint_path")
        }
        for checkpoint in self.checkpoint_root.glob("epoch_*"):
            if checkpoint.resolve() not in referenced:
                shutil.rmtree(checkpoint)

    def mark_complete(self, *, trained_through_epoch: int | None = None) -> None:
        manifest = json.loads(self.manifest_path.read_text())
        manifest["status"] = "complete"
        manifest["selectors"] = self.load_selectors()
        if trained_through_epoch is not None:
            manifest["trained_through_epoch"] = trained_through_epoch
        atomic_write_json(self.manifest_path, manifest)

    def update_runtime_summary(self, values: dict[str, Any]) -> None:
        manifest = json.loads(self.manifest_path.read_text())
        manifest.setdefault("runtime_summary", {}).update(values)
        atomic_write_json(self.manifest_path, manifest)

    def mark_paused(self, *, trained_through_epoch: int, rung: int) -> None:
        manifest = json.loads(self.manifest_path.read_text())
        manifest["status"] = "paused"
        manifest["trained_through_epoch"] = trained_through_epoch
        manifest["rung"] = rung
        manifest["selectors"] = self.load_selectors()
        atomic_write_json(self.manifest_path, manifest)

    def mark_screened_out(
        self,
        *,
        reason: str,
        promotion_manifest: str | None = None,
    ) -> None:
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("status") not in {"paused", "complete"}:
            raise ValueError("Only paused or complete runs can be marked screened_out")
        manifest["status"] = "screened_out"
        manifest["screened_out_reason"] = reason
        if promotion_manifest is not None:
            manifest["promotion_manifest"] = promotion_manifest
        atomic_write_json(self.manifest_path, manifest)

    def mark_failed(self, message: str) -> None:
        if not self.manifest_path.exists():
            return
        manifest = json.loads(self.manifest_path.read_text())
        manifest["status"] = "failed"
        manifest["error"] = message
        atomic_write_json(self.manifest_path, manifest)
