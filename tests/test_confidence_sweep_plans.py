import json
from pathlib import Path

from src.confidence_training.config import RunConfig
from src.confidence_training.sweep_plans import (
    build_continuation_plan,
    build_initial_linear_probe_plan,
    default_linear_probe_learning_rates,
)
from src.confidence_training.wandb_logging import linear_probe_project


MODELS = ("Olmo-3-7B-Instruct", "Qwen3-8B-non-thinking", "gpt-oss-20b")
DATASETS = ("triviaqa", "math", "mmlu")
LOSSES = ("mse", "bce", "ranknet")


def _pilots(root: Path) -> list[dict]:
    runs = []
    for model in MODELS:
        for dataset in DATASETS:
            for loss in LOSSES:
                lr = default_linear_probe_learning_rates(model, dataset, loss)[0]
                name = f"{model}__{dataset}__{loss}"
                run_dir = root / name
                run_dir.mkdir(parents=True)
                (run_dir / "history.jsonl").write_text(
                    json.dumps(
                        {
                            "kind": "epoch",
                            "epoch": 1,
                            "elapsed_seconds": 2.0,
                            "validation_mse": 0.1,
                            "validation_spearman_r": 0.5,
                        }
                    )
                    + "\n"
                )
                config = RunConfig(
                    run_name=name,
                    dataset_name=dataset,
                    model_name=model,
                    model_id="model-id",
                    adaptation_strategy="linear_probe",
                    loss=loss,
                    train_dir="train",
                    validation_dir="validation",
                    output_dir=str(run_dir),
                    learning_rate=lr,
                    batch_size=32,
                    epochs=1000,
                    seed=42,
                    standardize="gpt-oss" in model.casefold(),
                    device="cpu",
                )
                runs.append(
                    {
                        "status": "complete",
                        "config": config.to_dict(),
                        "run_dir": str(run_dir),
                    }
                )
    return runs


def test_initial_plan_has_five_lrs_per_arm_batch_and_nine_projects(tmp_path: Path) -> None:
    plan, configs = build_initial_linear_probe_plan(
        _pilots(tmp_path / "pilots"),
        output_root=tmp_path / "runs",
        config_root=tmp_path / "configs",
        max_epochs_by_batch={"8": 250, "16": 500, "32": 1000},
        stop_epochs_by_batch={"8": 25, "16": 50, "32": 100},
    )
    assert plan["reused_pilot_count"] == 27
    assert plan["job_count"] == 378
    assert len(configs) == 378
    represented = {
        (
            job["condition"]["model_name"],
            job["condition"]["dataset_name"],
            job["condition"]["loss"],
            job["condition"]["batch_size"],
        )
        for job in plan["jobs"]
    }
    # The 27 reused jobs supply the fifth LR in their batch-32 groups.
    assert len(represented) == 81
    all_projects = {
        value["wandb_project"]
        for _, value in configs
    } | {
        linear_probe_project(model, dataset) for model in MODELS for dataset in DATASETS
    }
    assert len(all_projects) == 9
    assert all(job["command"][:2] == ["uv", "run"] for job in plan["jobs"])


def test_continuation_plan_resumes_paused_and_skips_complete(tmp_path: Path) -> None:
    paused, complete = _pilots(tmp_path / "pilots")[:2]
    paused["status"] = "paused"
    paused["trained_through_epoch"] = 25
    paused["config"]["batch_size"] = 8
    paused["config"]["epochs"] = 250
    complete["status"] = "complete"
    promotion = {
        "promoted_run_dirs": [paused["run_dir"], complete["run_dir"]]
    }
    plan = build_continuation_plan(
        promotion,
        [paused, complete],
        stop_epochs_by_batch={"8": 75, "32": 300},
        rung=1,
    )
    assert plan["job_count"] == 1
    assert plan["already_complete"] == [complete["run_dir"]]
    command = plan["jobs"][0]["command"]
    assert "--resume" in command
    assert command[command.index("--stop-after-epoch") + 1] == "75"
