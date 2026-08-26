import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.confidence_training.config import RunConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("strategy", ["linear_probe", "lora", "full_weight"])
@pytest.mark.parametrize("loss", ["mse", "bce", "ranknet"])
def test_training_dry_run_for_every_arm(strategy: str, loss: str, tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/train_confidence_model.py"),
        "--run-name",
        "dry",
        "--dataset-name",
        "dataset",
        "--model-name",
        "model",
        "--model-id",
        "model-id",
        "--adaptation-strategy",
        strategy,
        "--loss",
        loss,
        "--train-dir",
        "train",
        "--validation-dir",
        "validation",
        "--output-dir",
        str(tmp_path / f"{strategy}-{loss}"),
        "--learning-rate",
        "0.001",
        "--batch-size",
        "8",
        "--epochs",
        "2",
        "--dry-run",
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    resolved = json.loads(result.stdout)
    assert resolved["adaptation_strategy"] == strategy
    assert resolved["loss"] == loss
    assert resolved["log_every_steps"] == (
        None if strategy == "linear_probe" else 10
    )
    assert not (tmp_path / f"{strategy}-{loss}").exists()


def test_json_config_then_cli_override(tmp_path: Path) -> None:
    config = {
        "run_name": "json",
        "dataset_name": "dataset",
        "model_name": "model",
        "model_id": "model-id",
        "adaptation_strategy": "linear_probe",
        "loss": "mse",
        "train_dir": "train",
        "validation_dir": "validation",
        "output_dir": str(tmp_path / "output"),
        "learning_rate": 0.01,
        "batch_size": 8,
        "epochs": 2,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/train_confidence_model.py"),
            "--config",
            str(path),
            "--learning-rate",
            "0.0002",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["learning_rate"] == 0.0002


def test_linear_probe_rejects_log_every_steps(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/train_confidence_model.py"),
            "--run-name",
            "invalid-step-logging",
            "--dataset-name",
            "dataset",
            "--model-name",
            "model",
            "--model-id",
            "model-id",
            "--adaptation-strategy",
            "linear_probe",
            "--loss",
            "mse",
            "--train-dir",
            "train",
            "--validation-dir",
            "validation",
            "--output-dir",
            str(tmp_path / "run"),
            "--log-every-steps",
            "10",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--log-every-steps cannot be used" in result.stderr


def test_preliminary_lora_launcher_resolves_gpt_protocol(tmp_path: Path) -> None:
    environment = {
        **__import__("os").environ,
        "DRY_RUN": "1",
        "MODEL_FILTER": "gpt-oss-20b",
        "DATASET_FILTER": "triviaqa",
        "LOSS_FILTER": "mse",
        "WANDB_MODE": "disabled",
        "OUTPUT_ROOT": str(tmp_path / "runs"),
    }
    result = subprocess.run(
        ["bash", "paper_results/confidence_training/run_preliminary_lora.sh"],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    resolved = json.loads(result.stdout)
    assert resolved["batch_size"] == 16
    assert resolved["learning_rate"] == 3e-5
    assert resolved["log_every_steps"] == 10
    assert resolved["lora_target_parameters"] == [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]
    assert resolved["train_chat_template_kwargs"]["template_date"] == "2026-08-23"
    assert resolved["validation_chat_template_kwargs"]["template_date"] == "2026-04-21"
    assert resolved["wandb_project"] == "lm-lora__gpt-oss-20b__triviaqa"


def test_wandb_backfill_dry_run_discovers_completed_lora_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "lora-run"
    run_dir.mkdir()
    config = RunConfig(
        run_name="lora-run",
        dataset_name="triviaqa",
        model_name="Olmo-3-7B-Instruct",
        model_id="model-id",
        adaptation_strategy="lora",
        loss="mse",
        train_dir="train",
        validation_dir="validation",
        output_dir=str(run_dir),
        learning_rate=3e-5,
        batch_size=16,
        epochs=3,
        wandb_project="lm-lora__Olmo-3-7B-Instruct__triviaqa",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"status": "complete", "config": config.to_dict()})
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/backfill_confidence_history_to_wandb.py"),
            "--runs-root",
            str(tmp_path),
            "--adaptation-strategy",
            "lora",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert len(plan["runs"]) == 1
    assert plan["runs"][0]["project"] == (
        "lm-lora__Olmo-3-7B-Instruct__triviaqa"
    )
