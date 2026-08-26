"""Hyperparameter selection, registry finalization, and safe pruning."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

from safetensors.torch import load_file

from .artifacts import atomic_write_json, sha256_path
from .isotonic import fit_isotonic_scores, save_isotonic


CONDITION_FIELDS = ("model_name", "dataset_name", "adaptation_strategy", "loss")
NON_PROTOCOL_FIELDS = {
    "run_name",
    "output_dir",
    "learning_rate",
    "batch_size",
    "epochs",
    "resume",
    "save_resume_state",
    "wandb_project",
    "wandb_entity",
    "wandb_mode",
    "log_every_steps",
}


def discover_runs(roots: str | Path | Sequence[str | Path]) -> list[dict[str, Any]]:
    """Discover and de-duplicate manifests below one or more roots."""

    root_values = [roots] if isinstance(roots, (str, Path)) else list(roots)
    paths = {
        path.resolve()
        for root in root_values
        for path in Path(root).rglob("run_manifest.json")
    }
    runs: list[dict[str, Any]] = []
    for path in sorted(paths):
        value = json.loads(path.read_text())
        value["manifest_path"] = str(path)
        value["run_dir"] = str(path.parent)
        runs.append(value)
    return runs


def _condition(config: dict[str, Any]) -> dict[str, Any]:
    return {field: config[field] for field in CONDITION_FIELDS}


def fixed_protocol(config: dict[str, Any]) -> dict[str, Any]:
    """Fields that must be identical within a sweep condition."""

    return {
        key: value
        for key, value in sorted(config.items())
        if key not in NON_PROTOCOL_FIELDS
    }


def _candidate_key(run: dict[str, Any], selector: str) -> tuple[Any, ...]:
    config = run["config"]
    entry = run["selectors"][selector]
    metrics = entry["metrics"]
    stable = (
        float(config["learning_rate"]),
        int(config["batch_size"]),
        str(config["run_name"]),
    )
    if selector == "mse":
        return (float(metrics["mse"]), -float(metrics["spearman_r"]), *stable)
    return (-float(metrics["spearman_r"]), float(metrics["mse"]), *stable)


def select_hyperparameters(
    runs: Sequence[dict[str, Any]],
    *,
    sweep_seed: int = 42,
) -> dict[str, Any]:
    incomplete = [
        run.get("run_dir", run.get("manifest_path", "<unknown>"))
        for run in runs
        if int(run.get("config", {}).get("seed", -1)) == sweep_seed
        and run.get("status") not in {"complete", "screened_out"}
    ]
    if incomplete:
        raise ValueError(f"Sweep contains incomplete seed-{sweep_seed} runs: {incomplete}")

    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        config = run["config"]
        if run.get("status") == "screened_out":
            continue
        if int(config["seed"]) != sweep_seed:
            continue
        condition = _condition(config)
        key = json.dumps(condition, sort_keys=True)
        groups.setdefault(key, []).append(run)

    entries: list[dict[str, Any]] = []
    for key in sorted(groups):
        condition = json.loads(key)
        group = groups[key]
        reference_protocol = fixed_protocol(group[0]["config"])
        drift = [
            run["run_dir"]
            for run in group[1:]
            if fixed_protocol(run["config"]) != reference_protocol
        ]
        if drift:
            raise ValueError(
                f"Fixed protocol fields differ within {condition}; mismatched runs: {drift}"
            )
        for selector in ("mse", "spearman_r"):
            candidates = [
                run
                for run in group
                if run.get("selectors", {}).get(selector, {}).get("checkpoint_path")
            ]
            if len(candidates) != len(group):
                raise ValueError(f"Missing {selector} checkpoint candidates for {condition}")
            selected = min(candidates, key=lambda run: _candidate_key(run, selector))
            config = selected["config"]
            selector_entry = selected["selectors"][selector]
            entries.append(
                {
                    "condition": {**condition, "selector": selector},
                    "selected_hyperparameters": {
                        "learning_rate": config["learning_rate"],
                        "batch_size": config["batch_size"],
                    },
                    "fixed_protocol": reference_protocol,
                    "training_budget": {
                        "epochs": int(config["epochs"]),
                        "trained_through_epoch": int(selected.get("trained_through_epoch", config["epochs"])),
                    },
                    "sweep_seed": sweep_seed,
                    "run_dir": selected["run_dir"],
                    "checkpoint_path": selector_entry["checkpoint_path"],
                    "epoch": selector_entry["epoch"],
                    "validation_metrics": selector_entry["metrics"],
                    "candidate_count": len(candidates),
                }
            )
    return {"selection_version": 2, "sweep_seed": sweep_seed, "entries": entries}


def _matching_final_run(
    runs: Sequence[dict[str, Any]],
    selection: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    condition = selection["condition"]
    hyperparameters = selection["selected_hyperparameters"]
    training_budget = selection.get("training_budget", {})
    matches = []
    for run in runs:
        if run.get("status") != "complete":
            continue
        config = run["config"]
        if int(config["seed"]) != seed:
            continue
        if any(config[field] != condition[field] for field in CONDITION_FIELDS):
            continue
        if float(config["learning_rate"]) != float(hyperparameters["learning_rate"]):
            continue
        if int(config["batch_size"]) != int(hyperparameters["batch_size"]):
            continue
        if training_budget and int(config["epochs"]) != int(training_budget["epochs"]):
            continue
        final_protocol = fixed_protocol(config)
        selected_protocol = dict(selection.get("fixed_protocol", final_protocol))
        # The seed is deliberately different while every other protocol field is frozen.
        final_protocol["seed"] = selected_protocol.get("seed", final_protocol.get("seed"))
        if final_protocol != selected_protocol:
            continue
        matches.append(run)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one protocol-matching final run for {condition}, seed={seed}; "
            f"found {len(matches)}"
        )
    return matches[0]


def finalize_registry(
    selection: dict[str, Any],
    final_runs: Sequence[dict[str, Any]],
    *,
    seeds: Sequence[int] = (42, 43, 44),
    expected_configurations: int | None = 162,
) -> dict[str, Any]:
    if len(set(seeds)) != len(seeds):
        raise ValueError("Final registry seeds must be unique")
    if expected_configurations is not None and len(selection["entries"]) != expected_configurations:
        raise ValueError(
            f"Expected {expected_configurations} selected configurations; "
            f"found {len(selection['entries'])}"
        )
    registry_entries: list[dict[str, Any]] = []
    for selected in selection["entries"]:
        selector = selected["condition"]["selector"]
        seed_entries: list[dict[str, Any]] = []
        for seed in seeds:
            run = _matching_final_run(final_runs, selected, seed)
            selector_state = run.get("selectors", {}).get(selector, {})
            if not selector_state.get("checkpoint_path"):
                raise ValueError(f"Final run lacks selected {selector} checkpoint: {run['run_dir']}")
            checkpoint = Path(selector_state["checkpoint_path"])
            validation_path = checkpoint / "validation.safetensors"
            values = load_file(validation_path)
            artifact = fit_isotonic_scores(
                values["scores"].tolist(),
                values["targets"].tolist(),
                metadata={
                    "selector": selector,
                    "seed": seed,
                    "checkpoint_path": str(checkpoint),
                    "validation_path": str(validation_path),
                },
            )
            isotonic_path = save_isotonic(artifact, checkpoint)
            seed_entries.append(
                {
                    "seed": seed,
                    "run_dir": run["run_dir"],
                    "scientific_config_hash": run.get("scientific_config_hash"),
                    "config": run["config"],
                    "provenance": run.get("provenance", {}),
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": sha256_path(checkpoint),
                    "epoch": selector_state["epoch"],
                    "validation_metrics": selector_state["metrics"],
                    "validation_path": str(validation_path),
                    "validation_sha256": sha256_path(validation_path),
                    "isotonic_path": str(isotonic_path),
                    "isotonic_sha256": sha256_path(isotonic_path),
                }
            )
        registry_entries.append(
            {
                "condition": selected["condition"],
                "selected_hyperparameters": selected["selected_hyperparameters"],
                "fixed_protocol": selected.get("fixed_protocol"),
                "seeds": seed_entries,
            }
        )
    registry: dict[str, Any] = {
        "registry_version": 1,
        "frozen": True,
        "seeds": list(seeds),
        "configuration_count": len(registry_entries),
        "entries": registry_entries,
    }
    registry["registry_digest"] = digest_json(registry)
    return registry


def digest_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_immutable_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {destination}")
    atomic_write_json(destination, value)


def referenced_checkpoints(*documents: dict[str, Any]) -> set[Path]:
    references: set[Path] = set()
    for document in documents:
        if "entries" not in document:
            continue
        if document.get("registry_version"):
            for entry in document["entries"]:
                for seed_entry in entry["seeds"]:
                    references.add(Path(seed_entry["checkpoint_path"]).resolve())
        else:
            for entry in document["entries"]:
                references.add(Path(entry["checkpoint_path"]).resolve())
    return references


def prune_unreferenced(
    runs_root: str | Path,
    references: Iterable[Path],
    *,
    apply: bool = False,
) -> list[str]:
    root = Path(runs_root).resolve()
    referenced = {path.resolve() for path in references}
    candidates: list[str] = []
    for checkpoint in sorted(root.rglob("checkpoints/epoch_*")):
        resolved = checkpoint.resolve()
        if root not in resolved.parents:
            raise ValueError(f"Checkpoint escapes runs root: {checkpoint}")
        if resolved in referenced:
            continue
        candidates.append(str(checkpoint))
        if apply:
            shutil.rmtree(checkpoint)
    return candidates
