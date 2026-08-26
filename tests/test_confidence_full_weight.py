import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.confidence_training.artifacts import RunArtifacts
from src.confidence_training.config import FULL_WEIGHT_MODEL_SPECS, RunConfig
from src.confidence_training.data import audit_dataset_folder
from src.confidence_training.models import (
    LLMScoreModel,
    build_optimizer,
    load_score_model,
    parameter_report,
    prepare_full_weight_backbone,
    save_score_model,
)
from src.confidence_training.wandb_logging import full_weight_project


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_prompt_folder(path: Path, suffix: str = "") -> Path:
    path.mkdir(parents=True)
    rows = [
        {
            "example_id": f"a{suffix}",
            "question": "Question A",
            "num_samples": 4,
            "num_correct": 1,
            "expected_accuracy": 0.25,
        },
        {
            "example_id": f"b{suffix}",
            "question": "Question B",
            "num_samples": 2,
            "num_correct": 2,
            "expected_accuracy": 1.0,
        },
    ]
    write_jsonl(path / "ground_truth.jsonl", rows)
    write_jsonl(
        path / "sampled.jsonl",
        [
            {"example_id": row["example_id"], "prompt": [{"role": "user", "content": row["question"]}]}
            for row in rows
        ],
    )
    return path


def full_config(output: Path, model_name: str = "tiny") -> RunConfig:
    config = RunConfig(
        run_name="tiny-full",
        dataset_name="triviaqa",
        model_name=model_name,
        model_id="tiny/model",
        adaptation_strategy="full_weight",
        loss="mse",
        train_dir="train",
        validation_dir="validation",
        output_dir=str(output),
        learning_rate=1e-5,
        batch_size=16,
        epochs=3,
        gradient_checkpointing=True,
        log_every_steps=10,
    )
    config.validate()
    return config


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=3)
        self.proj = nn.Linear(3, 3, dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask, return_dict=True, **kwargs):
        del attention_mask, return_dict, kwargs
        values = torch.nn.functional.one_hot(input_ids, num_classes=3).to(torch.bfloat16)
        return SimpleNamespace(last_hidden_state=self.proj(values))

    def save_pretrained(self, path, **kwargs):
        del kwargs
        destination = Path(path)
        destination.mkdir(parents=True)
        torch.save(self.state_dict(), destination / "tiny.pt")


def test_adapted_lm_audit_requires_only_ground_truth_and_prompts(tmp_path: Path) -> None:
    folder = make_prompt_folder(tmp_path / "prompt-only")
    adapted = audit_dataset_folder(folder, scope="adapted_lm", expected_count=2)
    assert adapted["ok"]
    assert adapted["audit_scope"] == "adapted_lm"
    assert adapted["ordered_data_hash"] == adapted["ground_truth_hash"]
    assert adapted["ordered_prompt_hash"] == adapted["prompt_hash"]
    probing = audit_dataset_folder(folder, scope="probing_artifacts")
    assert not probing["ok"]
    assert any("probing_dataset_cc" in error for error in probing["errors"])
    wrong_count = audit_dataset_folder(folder, scope="adapted_lm", expected_count=3)
    assert not wrong_count["ok"]


def test_full_weight_matrix_has_exact_olmo_qwen_twelve_entries() -> None:
    matrix = json.loads(
        (REPO_ROOT / "paper_results/confidence_training/full_weight_data_matrix.json").read_text()
    )
    assert matrix["audit_scope"] == "adapted_lm"
    assert len(matrix["entries"]) == 12
    assert {entry["model_name"] for entry in matrix["entries"]} == set(FULL_WEIGHT_MODEL_SPECS)
    assert all("gpt-oss" not in entry["path"] for entry in matrix["entries"])
    assert {(entry["dataset"], entry["role"]) for entry in matrix["entries"]} == {
        (dataset, role)
        for dataset in ("triviaqa", "math", "mmlu")
        for role in ("train", "validation")
    }


def test_full_weight_backbone_is_bf16_trainable_and_strict() -> None:
    backbone = TinyBackbone().float()
    count = prepare_full_weight_backbone(backbone, expected_parameters=12)
    assert count == 12
    assert all(parameter.requires_grad for parameter in backbone.parameters())
    assert {parameter.dtype for parameter in backbone.parameters()} == {torch.bfloat16}

    frozen = TinyBackbone()
    frozen.proj.bias.requires_grad_(False)
    with pytest.raises(ValueError, match="arrived with frozen"):
        prepare_full_weight_backbone(frozen)
    quantized = TinyBackbone()
    quantized.config.quantization_config = {"quant_method": "mxfp4"}
    with pytest.raises(ValueError, match="Quantized full-weight"):
        prepare_full_weight_backbone(quantized)
    with pytest.raises(ValueError, match="count drift"):
        prepare_full_weight_backbone(TinyBackbone(), expected_parameters=11)


def test_full_weight_parameter_and_optimizer_report_is_exact() -> None:
    model = LLMScoreModel(TinyBackbone())
    config = full_config(Path("unused"))
    optimizer = build_optimizer(model, config)
    report = parameter_report(model, optimizer, adaptation_strategy="full_weight")
    assert report["logical_backbone_parameters"] == 12
    assert report["trainable_backbone_parameters"] == 12
    assert report["total_by_component"]["head"] == 4
    assert report["parameters_by_dtype"] == {"torch.bfloat16": 12, "torch.float32": 4}
    assert report["optimizer_parameter_occurrences"] == len(list(model.parameters()))
    assert report["optimizer_duplicate_parameter_tensors"] == 0
    assert report["optimizer_frozen_parameter_count"] == 0
    assert report["optimizer_matches_trainable"]
    assert report["full_weight_trainable_backbone_matches_logical"]


def test_parameter_and_dtype_report_persist_in_resolved_config_and_manifest(
    tmp_path: Path,
) -> None:
    model = LLMScoreModel(TinyBackbone())
    config = full_config(tmp_path / "run")
    report = parameter_report(
        model, build_optimizer(model, config), adaptation_strategy="full_weight"
    )
    artifacts = RunArtifacts(config)
    artifacts.initialize({"parameter_report": report})
    resolved = json.loads((artifacts.root / "resolved_config.json").read_text())
    manifest = json.loads(artifacts.manifest_path.read_text())
    for value in (resolved["_runtime"], manifest):
        assert value["parameter_report"] == report
        assert value["actual_dtypes"] == report["parameters_by_dtype"]


def test_full_weight_config_rejects_gpt_oss() -> None:
    with pytest.raises(ValueError, match="excluded"):
        full_config(Path("unused"), model_name="gpt-oss-20b")


def test_full_weight_fake_hf_checkpoint_never_falls_back_to_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    checkpoint = run_root / "checkpoints" / "epoch_000001" / "model"
    (run_root / "tokenizer").mkdir(parents=True)
    model = LLMScoreModel(TinyBackbone())
    with torch.no_grad():
        model.regression_head.weight.fill_(0.25)
        model.regression_head.bias.fill_(-0.5)
    config = full_config(run_root)
    save_score_model(model, checkpoint, config)

    import transformers

    seen: dict[str, Path] = {}

    def load_backbone(path, **kwargs):
        del kwargs
        seen["backbone"] = Path(path)
        restored = TinyBackbone()
        restored.load_state_dict(torch.load(Path(path) / "tiny.pt", weights_only=True))
        return restored

    def load_tokenizer(path, **kwargs):
        del kwargs
        seen["tokenizer"] = Path(path)
        return object()

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", load_backbone)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    restored, _ = load_score_model(checkpoint, config, device=torch.device("cpu"))
    assert seen["backbone"] == checkpoint / "backbone"
    assert seen["tokenizer"] == run_root / "tokenizer"
    batch = {
        "input_ids": torch.tensor([[0, 1, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    torch.testing.assert_close(restored.forward_scores(**batch), model.forward_scores(**batch))


def decode_json_stream(value: str) -> list[dict]:
    decoder = json.JSONDecoder()
    rows = []
    index = 0
    while index < len(value):
        while index < len(value) and value[index].isspace():
            index += 1
        if index == len(value):
            break
        row, index = decoder.raw_decode(value, index)
        rows.append(row)
    return rows


def launcher(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "paper_results/confidence_training/run_preliminary_full_weight.sh"],
        cwd=REPO_ROOT,
        env={**os.environ, "DRY_RUN": "1", "WANDB_MODE": "disabled", **environment},
        capture_output=True,
        text=True,
    )


def test_full_weight_launcher_emits_exact_fixed_eighteen_arm_matrix(tmp_path: Path) -> None:
    result = launcher({"OUTPUT_ROOT": str(tmp_path / "does-not-exist")})
    assert result.returncode == 0, result.stderr
    rows = decode_json_stream(result.stdout)
    assert len(rows) == 18
    keys = [(row["model_name"], row["dataset_name"], row["loss"]) for row in rows]
    assert len(keys) == len(set(keys))
    assert keys == [
        (model, dataset, loss)
        for model in ("Olmo-3-7B-Instruct", "Qwen3-8B-non-thinking")
        for dataset in ("triviaqa", "math", "mmlu")
        for loss in ("mse", "bce", "ranknet")
    ]
    assert {row["wandb_project"] for row in rows} == {
        full_weight_project(model, dataset)
        for model in FULL_WEIGHT_MODEL_SPECS
        for dataset in ("triviaqa", "math", "mmlu")
    }
    for row in rows:
        assert row["adaptation_strategy"] == "full_weight"
        assert row["batch_size"] == 16
        assert row["learning_rate"] == 1e-5
        assert row["epochs"] == 3
        assert row["seed"] == 42
        assert row["gradient_checkpointing"] is True
        assert row["standardize"] is False
        assert row["num_workers"] == 0
        assert row["log_every_steps"] == 10
        assert row["save_resume_state"] is False
        assert row["resume"] is False
        assert row["wandb_entity"] == "brian-wu"
        assert not Path(row["output_dir"]).exists()


def test_full_weight_launcher_filters_and_rejects_conflicts(tmp_path: Path) -> None:
    selected = launcher(
        {
            "OUTPUT_ROOT": str(tmp_path / "filtered"),
            "MODEL_FILTER": "Qwen3-8B-non-thinking",
            "DATASET_FILTER": "math",
            "LOSS_FILTER": "ranknet",
        }
    )
    assert selected.returncode == 0, selected.stderr
    rows = decode_json_stream(selected.stdout)
    assert [(row["model_name"], row["dataset_name"], row["loss"]) for row in rows] == [
        ("Qwen3-8B-non-thinking", "math", "ranknet")
    ]
    rejected_gpt = launcher({"MODEL_FILTER": "gpt-oss-20b"})
    assert rejected_gpt.returncode != 0
    assert "unconditionally excluded" in rejected_gpt.stderr
    conflicting = launcher({"BATCH_SIZE": "8"})
    assert conflicting.returncode != 0
    assert "BATCH_SIZE=16" in conflicting.stderr


def test_full_weight_smoke_and_meta_preflight_dry_runs_have_no_side_effects(
    tmp_path: Path,
) -> None:
    smoke_output = tmp_path / "smoke"
    smoke = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/smoke_full_weight_capacity.py"),
            "--model-name",
            "Olmo-3-7B-Instruct",
            *(item for index in range(6) for item in ("--dataset-dir", f"missing-{index}")),
            "--output-dir",
            str(smoke_output),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert smoke.returncode == 0, smoke.stderr
    assert json.loads(smoke.stdout)["adaptation_strategy"] == "full_weight"
    assert not smoke_output.exists()

    preflight_output = tmp_path / "preflight.json"
    preflight = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/preflight_confidence_model.py"),
            "--model-name",
            "Qwen3-8B-non-thinking",
            "--model-id",
            "Qwen/Qwen3-8B",
            "--revision",
            str(FULL_WEIGHT_MODEL_SPECS["Qwen3-8B-non-thinking"]["revision"]),
            "--output",
            str(preflight_output),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert preflight.returncode == 0, preflight.stderr
    assert json.loads(preflight.stdout)["strategy"] == "full_weight"
    assert not preflight_output.exists()


def test_full_weight_backfill_dry_run_uses_project_and_deterministic_identity(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = full_config(run_dir, model_name="Olmo-3-7B-Instruct")
    config.dataset_name = "triviaqa"
    config.wandb_project = full_weight_project(config.model_name, config.dataset_name)
    config.validate()
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
            "full_weight",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert len(plan["runs"]) == 1
    assert plan["runs"][0]["project"] == full_weight_project(
        "Olmo-3-7B-Instruct", "triviaqa"
    )
    assert len(plan["runs"][0]["wandb_id"]) == 32
    assert not (tmp_path / "wandb_backfill_state.json").exists()


def make_synthetic_full_weight_tree(root: Path) -> None:
    for model_name, model_spec in FULL_WEIGHT_MODEL_SPECS.items():
        for dataset_name in ("triviaqa", "math", "mmlu"):
            for loss in ("mse", "bce", "ranknet"):
                run_name = (
                    f"{dataset_name}__{model_name}__full_weight__{loss}__"
                    "lr-1e-5__bs-16__seed-42"
                )
                run_dir = root / run_name
                checkpoint = run_dir / "checkpoints" / "epoch_000002"
                checkpoint.mkdir(parents=True)
                (checkpoint / "validation.safetensors").write_bytes(b"synthetic")
                (checkpoint / "checkpoint_manifest.json").write_text(
                    json.dumps({"checkpoint_version": 1, "model_sha256": "abc"})
                )
                config = RunConfig(
                    run_name=run_name,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    model_id=str(model_spec["model_id"]),
                    model_revision=str(model_spec["revision"]),
                    adaptation_strategy="full_weight",
                    loss=loss,
                    train_dir="train",
                    validation_dir="validation",
                    output_dir=str(run_dir),
                    learning_rate=1e-5,
                    batch_size=16,
                    epochs=3,
                    seed=42,
                    gradient_checkpointing=True,
                    num_workers=0,
                    log_every_steps=10,
                    save_resume_state=False,
                )
                config.validate()
                metrics = {"mse": 0.1, "spearman_r": 0.5}
                selector = {
                    "epoch": 2,
                    "value": metrics["mse"],
                    "checkpoint_path": str(checkpoint),
                    "validation_path": str(checkpoint / "validation.safetensors"),
                    "metrics": metrics,
                }
                manifest = {
                    "status": "complete",
                    "config": config.to_dict(),
                    "selectors": {"mse": selector, "spearman_r": selector},
                    "runtime_summary": {"wall_seconds": 7.5},
                }
                (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
                history = [
                    {"kind": "validation", "epoch": 0, "optimizer_steps": 0},
                    *[
                        {
                            "kind": "epoch",
                            "epoch": epoch,
                            "elapsed_seconds": float(epoch),
                            "optimizer_steps": epoch,
                        }
                        for epoch in (1, 2, 3)
                    ],
                ]
                write_jsonl(run_dir / "history.jsonl", history)


def test_full_weight_summary_accepts_exact_tree_and_rejects_missing_without_outputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    make_synthetic_full_weight_tree(root)
    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "summary.csv"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/summarize_preliminary_confidence_runs.py"),
        "--root",
        str(root),
        "--expected-runs",
        "18",
        "--output-json",
        str(output_json),
        "--output-csv",
        str(output_csv),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(output_json.read_text())
    assert summary["completed_runs"] == 18
    assert summary["integrity_ok"]
    assert len(summary["runs"]) == 18
    assert all("per_epoch_seconds" in row for row in summary["runs"])
    assert output_csv.is_file()

    next(root.glob("*/run_manifest.json")).unlink()
    failed_json = tmp_path / "failed-summary.json"
    failed_csv = tmp_path / "failed-summary.csv"
    failed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/summarize_preliminary_confidence_runs.py"),
            "--root",
            str(root),
            "--expected-runs",
            "18",
            "--output-json",
            str(failed_json),
            "--output-csv",
            str(failed_csv),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert not failed_json.exists()
    assert not failed_csv.exists()
