import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from src.confidence_training.artifacts import RunArtifacts
from src.confidence_training.collate import collate_linear
from src.confidence_training.config import RunConfig
from src.confidence_training.data import LinearFeatureDataset
from src.confidence_training.models import LinearScoreModel, build_optimizer
from src.confidence_training.selection import (
    discover_runs,
    finalize_registry,
    select_hyperparameters,
    write_immutable_json,
)
from src.confidence_training.training import set_global_seed, train_run


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_split(
    root: Path,
    *,
    prefix: str,
    targets: list[float],
) -> LinearFeatureDataset:
    root.mkdir(parents=True)
    ids = [f"{prefix}-{index}" for index in range(len(targets))]
    with (root / "ground_truth.jsonl").open("w") as ground_truth, (
        root / "sampled.jsonl"
    ).open("w") as sampled:
        for index, (example_id, target) in enumerate(zip(ids, targets, strict=True)):
            ground_truth.write(
                json.dumps(
                    {
                        "example_id": example_id,
                        "question": f"{prefix} question {index}",
                        "num_samples": 4,
                        "num_correct": int(target * 4),
                        "expected_accuracy": target,
                    }
                )
                + "\n"
            )
            sampled.write(
                json.dumps(
                    {
                        "example_id": example_id,
                        "sampled_id": 0,
                        "prompt": [{"role": "user", "content": f"prompt {example_id}"}],
                    }
                )
                + "\n"
            )
    probing = root / "probing_dataset_cc"
    probing.mkdir()
    features = torch.arange(len(targets), dtype=torch.float32).reshape(-1, 1)
    hidden = features[:, None, :].repeat(1, 2, 1)
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    save_file({"hidden_states": hidden}, probing / "hidden_states.safetensors")
    save_file({"targets": target_tensor}, probing / "targets.safetensors")
    (probing / "metadata.json").write_text(
        json.dumps(
            {
                "n_examples": len(ids),
                "n_layers": 2,
                "hidden_dim": 1,
                "example_ids": ids,
                "rendered_prompt_hash": f"rendered-{prefix}",
                "construction_hash": f"construction-{prefix}",
            }
        )
    )
    return LinearFeatureDataset(features, target_tensor, ids)


def _train(
    runs_root: Path,
    train_dir: Path,
    validation_dir: Path,
    train_dataset: LinearFeatureDataset,
    validation_dataset: LinearFeatureDataset,
    *,
    name: str,
    learning_rate: float,
    seed: int,
) -> None:
    set_global_seed(seed)
    config = RunConfig(
        run_name=name,
        dataset_name="tiny",
        model_name="tiny-model",
        model_id="tiny/model",
        adaptation_strategy="linear_probe",
        loss="mse",
        train_dir=str(train_dir),
        validation_dir=str(validation_dir),
        output_dir=str(runs_root / name),
        learning_rate=learning_rate,
        batch_size=2,
        epochs=2,
        seed=seed,
        device="cpu",
    )
    model = LinearScoreModel(1)
    optimizer = build_optimizer(model, config)
    artifacts = RunArtifacts(config)
    artifacts.initialize({"integration_test": True})
    train_run(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_collate=collate_linear,
        validation_collate=collate_linear,
        config=config,
        artifacts=artifacts,
    )


def test_tiny_sweep_registry_and_disjoint_evaluation(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"
    test_dir = tmp_path / "test"
    train_dataset = _write_split(train_dir, prefix="train", targets=[0.0, 0.25, 0.75, 1.0])
    validation_dataset = _write_split(
        validation_dir,
        prefix="validation",
        targets=[0.0, 0.25, 0.75, 1.0],
    )
    _write_split(test_dir, prefix="test", targets=[0.0, 0.25, 0.75, 1.0])

    sweep_root = tmp_path / "sweep"
    for index, learning_rate in enumerate((0.01, 0.001)):
        _train(
            sweep_root,
            train_dir,
            validation_dir,
            train_dataset,
            validation_dataset,
            name=f"sweep-{index}",
            learning_rate=learning_rate,
            seed=42,
        )
    selection = select_hyperparameters(discover_runs(sweep_root))
    assert len(selection["entries"]) == 2

    selected_rates = {
        float(entry["selected_hyperparameters"]["learning_rate"])
        for entry in selection["entries"]
    }
    final_root = tmp_path / "final"
    for seed in (42, 43, 44):
        for index, learning_rate in enumerate(sorted(selected_rates)):
            _train(
                final_root,
                train_dir,
                validation_dir,
                train_dataset,
                validation_dataset,
                name=f"final-{seed}-{index}",
                learning_rate=learning_rate,
                seed=seed,
            )
    registry = finalize_registry(
        selection,
        discover_runs(final_root),
        expected_configurations=2,
    )
    registry_path = tmp_path / "registry.json"
    write_immutable_json(registry_path, registry)

    output_dir = tmp_path / "evaluation"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/evaluate_confidence_model.py"),
            "--registry",
            str(registry_path),
            "--model-name",
            "tiny-model",
            "--dataset-name",
            "tiny",
            "--adaptation-strategy",
            "linear_probe",
            "--loss",
            "mse",
            "--selector",
            "mse",
            "--seed",
            "42",
            "--test-dir",
            str(test_dir),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert set(metrics) >= {"raw", "calibrated", "n_examples"}
    rows = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert set(rows[0]) >= {
        "score",
        "raw_confidence",
        "isotonic_confidence",
        "num_samples",
        "num_correct",
        "expected_accuracy",
    }

