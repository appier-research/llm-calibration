import json
from pathlib import Path

import pytest
import torch

from src.confidence_training.artifacts import RunArtifacts, atomic_write_json
from src.confidence_training.collate import collate_linear
from src.confidence_training.config import RunConfig
from src.confidence_training.control import TrainingControl
from src.confidence_training.data import LinearFeatureDataset
from src.confidence_training.models import LinearScoreModel, build_optimizer
from src.confidence_training.promotion import apply_screening, promote_runs
from src.confidence_training.scheduler import lpt_assignment, run_plan
from src.confidence_training.selection import discover_runs, select_hyperparameters
from src.confidence_training.training import set_global_seed, train_run
from src.confidence_training.wandb_logging import (
    WandbLogger,
    deterministic_wandb_id,
    linear_probe_project,
    normalize_history_row,
)


def _config(root: Path, *, resume: bool = False, epochs: int = 4, lr: float = 0.01) -> RunConfig:
    return RunConfig(
        run_name="same-scientific-run",
        dataset_name="triviaqa",
        model_name="Qwen3-8B-non-thinking",
        model_id="model-id",
        adaptation_strategy="linear_probe",
        loss="mse",
        train_dir="train",
        validation_dir="validation",
        output_dir=str(root),
        learning_rate=lr,
        batch_size=2,
        epochs=epochs,
        seed=42,
        device="cpu",
        resume=resume,
    )


def _dataset() -> LinearFeatureDataset:
    return LinearFeatureDataset(
        torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
        torch.tensor([0.0, 0.25, 0.75, 1.0]),
        ["a", "b", "c", "d"],
    )


def _train(config: RunConfig, control: TrainingControl) -> tuple[torch.nn.Module, dict]:
    model = LinearScoreModel(1)
    optimizer = build_optimizer(model, config)
    artifacts = RunArtifacts(config)
    artifacts.initialize({"test": True})
    manifest = train_run(
        model=model,
        optimizer=optimizer,
        train_dataset=_dataset(),
        validation_dataset=_dataset(),
        train_collate=collate_linear,
        validation_collate=collate_linear,
        config=config,
        artifacts=artifacts,
        control=control,
    )
    return model, manifest


def test_paused_resume_is_exact_at_epoch_boundary(tmp_path: Path) -> None:
    set_global_seed(7)
    uninterrupted_model, uninterrupted = _train(
        _config(tmp_path / "full"), TrainingControl()
    )
    set_global_seed(7)
    _, paused = _train(
        _config(tmp_path / "resumed"), TrainingControl(stop_after_epoch=2, rung=0)
    )
    assert paused["status"] == "paused"
    assert paused["trained_through_epoch"] == 2
    assert (tmp_path / "resumed" / "last.pt").exists()

    resumed_config = _config(tmp_path / "resumed", resume=True)
    resumed_model, resumed = _train(
        resumed_config, TrainingControl(stop_after_epoch=4, rung=1)
    )
    assert resumed["status"] == "complete"
    for name, value in uninterrupted_model.state_dict().items():
        torch.testing.assert_close(value, resumed_model.state_dict()[name], rtol=0, atol=0)
    for selector in ("mse", "spearman_r"):
        for field in ("epoch", "value", "metrics"):
            assert uninterrupted["selectors"][selector][field] == resumed["selectors"][selector][field]
    rows = [
        json.loads(line)
        for line in (tmp_path / "resumed" / "history.jsonl").read_text().splitlines()
    ]
    assert [row["epoch"] for row in rows] == [0, 1, 2, 3, 4]
    assert [row["optimizer_steps"] for row in rows] == [0, 2, 4, 6, 8]


def _write_promotion_run(
    root: Path,
    name: str,
    *,
    lr: float,
    values: list[tuple[float, float]],
    status: str = "paused",
    epochs: int = 4,
) -> dict:
    config = _config(root / name, epochs=epochs, lr=lr).to_dict()
    config["run_name"] = name
    manifest = {
        "status": status,
        "config": config,
        "selectors": {},
        "run_dir": str((root / name).resolve()),
        "trained_through_epoch": len(values),
    }
    (root / name).mkdir(parents=True)
    atomic_write_json(root / name / "run_manifest.json", manifest)
    with (root / name / "history.jsonl").open("w") as handle:
        handle.write(json.dumps({"kind": "validation", "epoch": 0, "mse": 1, "spearman_r": 0}) + "\n")
        for epoch, (mse, spearman) in enumerate(values, 1):
            handle.write(
                json.dumps(
                    {
                        "kind": "epoch",
                        "epoch": epoch,
                        "validation_mse": mse,
                        "validation_spearman_r": spearman,
                    }
                )
                + "\n"
            )
    return manifest


def test_dual_policy_promotion_uses_history_prefix_and_screens_union(tmp_path: Path) -> None:
    mse_run = _write_promotion_run(
        tmp_path, "mse", lr=1e-3, values=[(0.1, 0.2), (0.09, 0.1), (0.01, 0.9)]
    )
    spearman_run = _write_promotion_run(
        tmp_path, "spearman", lr=2e-3, values=[(0.2, 0.8), (0.19, 0.85), (0.3, 0.1)]
    )
    loser = _write_promotion_run(
        tmp_path, "loser", lr=3e-3, values=[(0.3, 0.1), (0.25, 0.15)]
    )
    promotion = promote_runs(
        [mse_run, spearman_run, loser], cutoff_epoch=2, rung=0
    )
    assert set(promotion["promoted_run_dirs"]) == {
        mse_run["run_dir"],
        spearman_run["run_dir"],
    }
    assert promotion["screened_out_run_dirs"] == [loser["run_dir"]]
    changed = apply_screening(promotion, promotion_path=tmp_path / "promotion.json")
    assert changed == [loser["run_dir"]]
    screened = json.loads((tmp_path / "loser" / "run_manifest.json").read_text())
    assert screened["status"] == "screened_out"


def _selectable_run(root: Path, name: str, *, epochs: int, status: str = "complete") -> dict:
    run = _write_promotion_run(
        root, name, lr=1e-3 if epochs == 4 else 2e-3, values=[(0.1, 0.5)], status=status, epochs=epochs
    )
    checkpoint = root / name / "checkpoints" / "epoch_000001"
    checkpoint.mkdir(parents=True)
    state = {
        "epoch": 1,
        "checkpoint_path": str(checkpoint),
        "metrics": {"mse": 0.1, "spearman_r": 0.5},
    }
    run["selectors"] = {"mse": state, "spearman_r": state}
    atomic_write_json(root / name / "run_manifest.json", run)
    return run


def test_selection_combines_roots_and_allows_variable_epoch_budgets(tmp_path: Path) -> None:
    first = _selectable_run(tmp_path / "first", "a", epochs=4)
    _selectable_run(tmp_path / "second", "b", epochs=8, status="screened_out")
    discovered = discover_runs([tmp_path / "first", tmp_path / "second"])
    selection = select_hyperparameters(discovered)
    assert len(selection["entries"]) == 2
    assert all(entry["run_dir"] == first["run_dir"] for entry in selection["entries"])
    assert all(entry["training_budget"]["epochs"] == 4 for entry in selection["entries"])


class _FakeRun:
    def __init__(self) -> None:
        self.logged = []
        self.summary = {}
        self.metrics = []
        self.finished = False

    def log(self, value, step=None):
        self.logged.append((value, step))

    def define_metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))

    def finish(self):
        self.finished = True


class _FakeWandb:
    def __init__(self) -> None:
        self.kwargs = None
        self.run = _FakeRun()

    def init(self, **kwargs):
        self.kwargs = kwargs
        return self.run


def test_wandb_namespaces_projects_and_resume_id_are_deterministic(tmp_path: Path) -> None:
    config = _config(tmp_path / "run")
    fake = _FakeWandb()
    logger = WandbLogger.create(config, config.to_dict(), wandb_module=fake, mode="offline")
    assert logger is not None
    assert fake.kwargs["project"] == linear_probe_project(config.model_name, config.dataset_name)
    assert fake.kwargs["id"] == deterministic_wandb_id(config)
    assert fake.kwargs["resume"] == "allow"
    logger.log(
        {
            "kind": "epoch",
            "epoch": 2,
            "optimizer_steps": 10,
            "rung": 1,
            "train_loss": 0.2,
            "gradient_norm_mean": 0.3,
            "elapsed_seconds": 1.5,
            "validation_mse": 0.1,
            "validation_spearman_r": 0.4,
        }
    )
    value, step = fake.run.logged[0]
    assert step is None
    assert value["train/epoch_loss"] == 0.2
    assert value["validation/mse"] == 0.1
    assert value["rung"] == 1
    assert normalize_history_row({"kind": "validation", "epoch": 0, "mse": 0.4})["validation/mse"] == 0.4


def test_adapted_model_step_logging_records_cadence_and_final_interval(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "step-logs", epochs=1)
    config.adaptation_strategy = "full_weight"
    config.log_every_steps = 3
    config.validate()
    _train(config, TrainingControl())

    rows = [
        json.loads(line)
        for line in (tmp_path / "step-logs" / "history.jsonl").read_text().splitlines()
    ]
    step_rows = [row for row in rows if row["kind"] == "train_step"]
    assert len(step_rows) == 1
    assert step_rows[0]["epoch_step"] == 2
    assert step_rows[0]["optimizer_steps"] == 2
    assert step_rows[0]["examples_seen_in_epoch"] == 4
    assert step_rows[0]["interval_examples"] == 4
    assert step_rows[0]["learning_rate"] == config.learning_rate
    assert step_rows[0]["interval_seconds"] >= 0
    normalized = normalize_history_row(step_rows[0])
    assert normalized["train/loss"] == step_rows[0]["interval_train_loss"]
    assert normalized["train/gradient_norm"] == (
        step_rows[0]["interval_gradient_norm_mean"]
    )
    assert [row["kind"] for row in rows] == [
        "validation",
        "train_step",
        "epoch",
    ]


def test_lpt_scheduler_balances_and_defers_jobs_that_cannot_fit(tmp_path: Path) -> None:
    jobs = [
        {"job_id": "long", "command": ["uv", "run", "true"], "predicted_seconds": 9},
        {"job_id": "medium", "command": ["uv", "run", "true"], "predicted_seconds": 5},
        {"job_id": "short", "command": ["uv", "run", "true"], "predicted_seconds": 4},
    ]
    lanes = lpt_assignment(jobs, max_workers=2)
    assert [lane["predicted_seconds"] for lane in lanes] == [9.0, 9.0]
    called = []

    def forbidden_popen(*args, **kwargs):
        called.append(args)
        raise AssertionError("A job beyond the deadline must not launch")

    state = run_plan(
        {"plan_version": 1, "jobs": jobs},
        state_path=tmp_path / "state.json",
        log_dir=tmp_path / "logs",
        max_workers=2,
        deadline_seconds=3,
        reserve_seconds=1,
        popen_factory=forbidden_popen,
    )
    assert not called
    assert {value["status"] for value in state["jobs"].values()} == {"deferred"}
