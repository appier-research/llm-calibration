import json
from pathlib import Path

import pytest
import torch

from src.confidence_training.artifacts import RunArtifacts
from src.confidence_training.collate import collate_linear
from src.confidence_training.config import RunConfig
from src.confidence_training.data import LinearFeatureDataset
from src.confidence_training.models import LinearScoreModel, build_optimizer
from src.confidence_training.training import train_run


class TrainingModeTrackingLinearModel(LinearScoreModel):
    def __init__(self) -> None:
        super().__init__(1)
        self.forward_training_modes: list[bool] = []

    def forward_scores(self, features: torch.Tensor) -> torch.Tensor:
        self.forward_training_modes.append(self.training)
        return super().forward_scores(features)


def test_tiny_training_validates_step_zero_and_epoch_boundaries(tmp_path: Path) -> None:
    config = RunConfig(
        run_name="tiny",
        dataset_name="dataset",
        model_name="model",
        model_id="model-id",
        adaptation_strategy="linear_probe",
        loss="mse",
        train_dir="train",
        validation_dir="validation",
        output_dir=str(tmp_path / "run"),
        learning_rate=0.01,
        batch_size=2,
        epochs=2,
        device="cpu",
    )
    config.validate()
    features = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    targets = torch.tensor([0.0, 0.25, 0.75, 1.0])
    dataset = LinearFeatureDataset(features, targets, ["a", "b", "c", "d"])
    model = LinearScoreModel(1)
    optimizer = build_optimizer(model, config)
    artifacts = RunArtifacts(config)
    artifacts.initialize({"test": True})
    manifest = train_run(
        model=model,
        optimizer=optimizer,
        train_dataset=dataset,
        validation_dataset=dataset,
        train_collate=collate_linear,
        validation_collate=collate_linear,
        config=config,
        artifacts=artifacts,
    )
    assert manifest["status"] == "complete"
    rows = [json.loads(line) for line in artifacts.history_path.read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [0, 1, 2]
    assert all(row["kind"] == "epoch" for row in rows[1:])
    selectors = json.loads(artifacts.selector_path.read_text())
    assert selectors["mse"]["epoch"] in (1, 2)
    assert selectors["spearman_r"]["epoch"] in (1, 2)


def test_step_validation_is_independent_and_preserves_training_mode(
    tmp_path: Path,
) -> None:
    config = RunConfig(
        run_name="step-validation",
        dataset_name="dataset",
        model_name="model",
        model_id="model-id",
        adaptation_strategy="linear_probe",
        loss="mse",
        train_dir="train",
        validation_dir="validation",
        output_dir=str(tmp_path / "run"),
        learning_rate=0.01,
        batch_size=1,
        epochs=1,
        device="cpu",
        log_every_steps=None,
        validate_every_k_steps=2,
    )
    config.validate()
    features = torch.arange(6, dtype=torch.float32).reshape(-1, 1)
    targets = torch.linspace(0.0, 1.0, 6)
    dataset = LinearFeatureDataset(
        features, targets, [str(index) for index in range(6)]
    )
    model = TrainingModeTrackingLinearModel()
    artifacts = RunArtifacts(config)
    artifacts.initialize({"test": True})

    train_run(
        model=model,
        optimizer=build_optimizer(model, config),
        train_dataset=dataset,
        validation_dataset=dataset,
        train_collate=collate_linear,
        validation_collate=collate_linear,
        config=config,
        artifacts=artifacts,
    )

    rows = [json.loads(line) for line in artifacts.history_path.read_text().splitlines()]
    validation_steps = [
        row["optimizer_steps"] for row in rows if row["kind"] == "validation"
    ]
    assert validation_steps == [0, 2, 4, 6]
    assert model.forward_training_modes.count(True) == 6
    assert model.forward_training_modes.count(False) == 24
    assert model.training


@pytest.mark.parametrize("value", [0, -1])
def test_validation_interval_must_be_positive(value: int) -> None:
    config = RunConfig(validate_every_k_steps=value)
    with pytest.raises(ValueError, match="validate_every_k_steps must be positive"):
        config.validate(require_paths=False)

