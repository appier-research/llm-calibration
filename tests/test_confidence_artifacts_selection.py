import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from src.confidence_training.artifacts import RunArtifacts
from src.confidence_training.config import RunConfig
from src.confidence_training.isotonic import apply_isotonic, fit_isotonic_scores
from src.confidence_training.selection import (
    finalize_registry,
    prune_unreferenced,
    referenced_checkpoints,
    select_hyperparameters,
    write_immutable_json,
)


def make_config(root: Path, *, run_name: str = "run", lr: float = 1e-3, batch: int = 8, seed: int = 42) -> RunConfig:
    config = RunConfig(
        run_name=run_name,
        dataset_name="triviaqa",
        model_name="model",
        model_id="model-id",
        adaptation_strategy="linear_probe",
        loss="mse",
        train_dir="train",
        validation_dir="validation",
        output_dir=str(root),
        learning_rate=lr,
        batch_size=batch,
        epochs=3,
        seed=seed,
    )
    config.validate()
    return config


def saver(path: Path) -> None:
    path.mkdir()
    torch.save({"value": 1}, path / "model.pt")


def test_dual_checkpoint_tracking_deduplicates_and_keeps_earliest_ties(tmp_path: Path) -> None:
    artifacts = RunArtifacts(make_config(tmp_path / "run"))
    artifacts.initialize({"test": True})
    assert artifacts.improvements(0, {"mse": 0.1, "spearman_r": 0.9}) == set()
    ids = ["a", "b"]
    tensors = torch.tensor([0.0, 1.0])
    metrics1 = {"mse": 0.2, "spearman_r": 0.1, "mae": 0.3, "rmse": 0.4, "pearson_r": 0.1, "ece": 0.2}
    improved = artifacts.improvements(1, metrics1)
    artifacts.save_improved_checkpoint(
        epoch=1,
        metrics=metrics1,
        example_ids=ids,
        scores=tensors,
        probabilities=torch.sigmoid(tensors),
        targets=tensors,
        improved=improved,
        model_saver=saver,
    )
    assert len(list(artifacts.checkpoint_root.glob("epoch_*"))) == 1
    assert artifacts.improvements(2, metrics1) == set()
    metrics2 = {**metrics1, "mse": 0.1, "spearman_r": 0.05}
    artifacts.save_improved_checkpoint(
        epoch=2,
        metrics=metrics2,
        example_ids=ids,
        scores=tensors,
        probabilities=torch.sigmoid(tensors),
        targets=tensors,
        improved=artifacts.improvements(2, metrics2),
        model_saver=saver,
    )
    assert len(list(artifacts.checkpoint_root.glob("epoch_*"))) == 2
    metrics3 = {**metrics1, "mse": 0.15, "spearman_r": 0.2}
    artifacts.save_improved_checkpoint(
        epoch=3,
        metrics=metrics3,
        example_ids=ids,
        scores=tensors,
        probabilities=torch.sigmoid(tensors),
        targets=tensors,
        improved=artifacts.improvements(3, metrics3),
        model_saver=saver,
    )
    assert sorted(path.name for path in artifacts.checkpoint_root.glob("epoch_*")) == [
        "epoch_000002",
        "epoch_000003",
    ]


def fake_run(root: Path, *, name: str, lr: float, batch: int, mse: float, spearman: float, seed: int = 42) -> dict:
    checkpoint = root / name / "checkpoints" / "epoch_000001"
    checkpoint.mkdir(parents=True)
    metrics = {"mse": mse, "spearman_r": spearman}
    config = make_config(root / name, run_name=name, lr=lr, batch=batch, seed=seed).to_dict()
    selector = {
        "epoch": 1,
        "checkpoint_path": str(checkpoint),
        "validation_path": str(checkpoint / "validation.safetensors"),
        "metrics": metrics,
    }
    return {
        "status": "complete",
        "config": config,
        "selectors": {"mse": selector, "spearman_r": selector},
        "run_dir": str(root / name),
    }


def test_hyperparameter_selection_uses_secondary_and_stable_ties(tmp_path: Path) -> None:
    runs = [
        fake_run(tmp_path, name="b", lr=2e-4, batch=16, mse=0.1, spearman=0.4),
        fake_run(tmp_path, name="a", lr=1e-4, batch=8, mse=0.1, spearman=0.5),
    ]
    selected = select_hyperparameters(runs)
    assert len(selected["entries"]) == 2
    assert all(entry["run_dir"].endswith("a") for entry in selected["entries"])


def test_isotonic_v2_accepts_scores_and_v1_remains_supported() -> None:
    artifact = fit_isotonic_scores([-5.0, 0.0, 4.0], [0.0, 0.4, 1.0])
    assert artifact["artifact_version"] == 2
    calibrated = apply_isotonic([-100.0, 100.0], artifact)
    assert calibrated.tolist() == [0.0, 1.0]
    legacy = {
        "artifact_version": 1,
        "X_thresholds_": [0.0, 1.0],
        "y_thresholds_": [0.2, 0.8],
    }
    assert apply_isotonic([0.5], legacy).tolist() == [0.5]


def test_registry_finalization_and_immutable_write(tmp_path: Path) -> None:
    selection_run = fake_run(tmp_path, name="sweep", lr=1e-4, batch=8, mse=0.1, spearman=0.5)
    selection = select_hyperparameters([selection_run])
    # Keep one selector entry so the fixture represents one logical configuration.
    selection["entries"] = [selection["entries"][0]]
    final_runs = []
    for seed in (42, 43, 44):
        run = fake_run(tmp_path, name=f"final-{seed}", lr=1e-4, batch=8, mse=0.1, spearman=0.5, seed=seed)
        checkpoint = Path(run["selectors"]["mse"]["checkpoint_path"])
        save_file(
            {
                "scores": torch.tensor([-1.0, 1.0]),
                "probabilities": torch.sigmoid(torch.tensor([-1.0, 1.0])),
                "targets": torch.tensor([0.0, 1.0]),
            },
            checkpoint / "validation.safetensors",
        )
        (checkpoint / "model").mkdir()
        (checkpoint / "model" / "placeholder").write_text("model")
        final_runs.append(run)
    registry = finalize_registry(selection, final_runs, expected_configurations=1)
    assert registry["configuration_count"] == 1
    assert len(registry["entries"][0]["seeds"]) == 3
    path = tmp_path / "registry.json"
    write_immutable_json(path, registry)
    with pytest.raises(FileExistsError):
        write_immutable_json(path, registry)


def test_pruning_is_dry_by_default_and_scoped(tmp_path: Path) -> None:
    keep = tmp_path / "runs" / "a" / "checkpoints" / "epoch_000001"
    remove = tmp_path / "runs" / "b" / "checkpoints" / "epoch_000002"
    keep.mkdir(parents=True)
    remove.mkdir(parents=True)
    document = {"selection_version": 1, "entries": [{"checkpoint_path": str(keep)}]}
    candidates = prune_unreferenced(
        tmp_path / "runs", referenced_checkpoints(document), apply=False
    )
    assert candidates == [str(remove)]
    assert remove.exists()
    prune_unreferenced(tmp_path / "runs", referenced_checkpoints(document), apply=True)
    assert keep.exists() and not remove.exists()

