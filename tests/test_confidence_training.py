import json
from pathlib import Path

import torch

from src.confidence_training.artifacts import RunArtifacts
from src.confidence_training.collate import collate_linear
from src.confidence_training.config import RunConfig
from src.confidence_training.data import LinearFeatureDataset
from src.confidence_training.models import LinearScoreModel, build_optimizer
from src.confidence_training.training import train_run


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

