from pathlib import Path

import pytest

from src.confidence_training.config import RunConfig
from src.confidence_training.selection import select_hyperparameters


def _run(
    root: Path,
    *,
    name: str,
    status: str = "complete",
    momentum: float = 0.9,
) -> dict:
    checkpoint = root / name / "checkpoints" / "epoch_000001"
    checkpoint.mkdir(parents=True)
    config = RunConfig(
        run_name=name,
        dataset_name="triviaqa",
        model_name="model",
        model_id="model-id",
        adaptation_strategy="linear_probe",
        loss="mse",
        train_dir="train",
        validation_dir="validation",
        output_dir=str(root / name),
        learning_rate=1e-3,
        batch_size=8,
        epochs=3,
        seed=42,
        sgd_momentum=momentum,
    ).to_dict()
    selector = {
        "epoch": 1,
        "checkpoint_path": str(checkpoint),
        "metrics": {"mse": 0.1, "spearman_r": 0.5},
    }
    return {
        "status": status,
        "config": config,
        "selectors": {"mse": selector, "spearman_r": selector},
        "run_dir": str(root / name),
    }


def test_selection_rejects_incomplete_sweep_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="incomplete"):
        select_hyperparameters([_run(tmp_path, name="running", status="running")])


def test_selection_rejects_fixed_protocol_drift(tmp_path: Path) -> None:
    runs = [
        _run(tmp_path, name="expected", momentum=0.9),
        _run(tmp_path, name="drift", momentum=0.8),
    ]
    with pytest.raises(ValueError, match="Fixed protocol"):
        select_hyperparameters(runs)

