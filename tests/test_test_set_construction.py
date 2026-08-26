import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import scripts.launch_test_sampling as test_launcher

from src.data.base import DataExample
from src.data.test_audit import audit_test_output
from src.data.test_sampling import (
    aggregate_ground_truth,
    atomic_write_json,
    atomic_write_jsonl,
    ensure_sampling_state,
    ensure_verification_contract,
    generation_contract,
    validate_sample_rows,
)
from src.data.test_selection import (
    build_selection_manifest,
    load_selection_manifest,
    write_immutable_manifest,
)
from src.data.vllm_preflight import validate_non_thinking_responses


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_exclusion(folder: Path, example_id: str, question: str) -> dict[str, str]:
    write_jsonl(
        folder / "ground_truth.jsonl",
        [
            {
                "example_id": example_id,
                "question": question,
                "num_samples": 1,
                "num_correct": 1,
                "expected_accuracy": 1.0,
            }
        ],
    )
    return {"role": "train", "model_name": folder.name, "path": str(folder)}


def make_manifest(tmp_path: Path, dataset_name: str = "triviaqa") -> dict:
    source = [
        DataExample(id="excluded-id", question="Unique", answer=["x"]),
        DataExample(id="different-id", question="  Duplicate   question ", answer=["y"]),
        DataExample(id="a", question="Question A", answer=["A"]),
        DataExample(id="b", question="Question B", answer=["B"]),
        DataExample(id="c", question="Question C", answer=["C"]),
    ]
    exclusions = [
        make_exclusion(tmp_path / "train-a", "excluded-id", "Elsewhere"),
        make_exclusion(tmp_path / "valid-b", "other-id", "Duplicate question"),
    ]
    return build_selection_manifest(
        dataset_name=dataset_name,
        test_name=f"{dataset_name}-test2",
        source_examples=source,
        source={"fixture": True},
        exclusion_entries=exclusions,
        seed=42,
        size=2,
    )


def sample_row(manifest: dict, example_id: str, sampled_id: int, *, judge=None) -> dict:
    selected = {row["example_id"]: row for row in manifest["selected_examples"]}
    string_correct = sampled_id % 2
    return {
        "example_id": example_id,
        "sampled_id": sampled_id,
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "prompt": [{"role": "user", "content": selected[example_id]["question"]}],
        "response": "answer",
        "parsed_response": "answer",
        "ground_truth_answer": selected[example_id]["answer"],
        "string_match_correctness": string_correct,
        "judge_correctness": judge,
        "correctness": judge if judge is not None else string_correct,
    }


def test_union_exclusion_is_by_id_and_canonical_question_and_deterministic(tmp_path: Path) -> None:
    first = make_manifest(tmp_path)
    second = make_manifest(tmp_path)
    assert first == second
    selected_ids = {row["example_id"] for row in first["selected_examples"]}
    assert "excluded-id" not in selected_ids
    assert "different-id" not in selected_ids
    assert first["selection"]["seed"] == 42


def test_manifest_is_immutable_and_detects_tampering(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    path = tmp_path / "manifest.json"
    write_immutable_manifest(path, manifest)
    with pytest.raises(FileExistsError):
        write_immutable_manifest(path, manifest)
    payload = json.loads(path.read_text())
    payload["selected_examples"][0]["question"] = "drift"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        load_selection_manifest(path)


def test_resume_contract_rejects_config_and_prompt_drift(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    prompts = {
        row["example_id"]: [{"role": "user", "content": row["question"]}]
        for row in manifest["selected_examples"]
    }
    contract = generation_contract(
        selection_manifest=manifest,
        model_id="model",
        k=2,
        max_tokens=8000,
        temperature=0.7,
        top_p=0.8,
        prompt_messages=prompts,
    )
    state_path = tmp_path / "sampling_state.json"
    ensure_sampling_state(state_path, contract)
    ensure_sampling_state(state_path, contract)
    changed = dict(contract)
    changed["max_tokens"] = 7999
    with pytest.raises(ValueError, match="generation contract"):
        ensure_sampling_state(state_path, changed)
    ensure_verification_contract(
        state_path,
        stage="judge",
        contract={"judge_model_id": "judge-a", "judge_base_url": "http://a"},
    )
    with pytest.raises(ValueError, match="verification contract"):
        ensure_verification_contract(
            state_path,
            stage="judge",
            contract={"judge_model_id": "judge-b", "judge_base_url": "http://b"},
        )


def test_resume_rows_reject_membership_duplicate_and_prompt_drift(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    ids = [row["example_id"] for row in manifest["selected_examples"]]
    prompts = {
        row["example_id"]: [{"role": "user", "content": row["question"]}]
        for row in manifest["selected_examples"]
    }
    rows = [sample_row(manifest, ids[0], 0)]
    validate_sample_rows(rows, selection_manifest=manifest, k=2, prompt_messages=prompts, require_complete=False, require_judge=False)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_sample_rows(rows + rows, selection_manifest=manifest, k=2, prompt_messages=prompts, require_complete=False, require_judge=False)
    bad = dict(rows[0])
    bad["example_id"] = "outside"
    with pytest.raises(ValueError, match="outside immutable"):
        validate_sample_rows([bad], selection_manifest=manifest, k=2, prompt_messages={**prompts, "outside": bad["prompt"]}, require_complete=False, require_judge=False)
    bad = dict(rows[0])
    bad["prompt"] = [{"role": "user", "content": "drift"}]
    with pytest.raises(ValueError, match="Prompt drift"):
        validate_sample_rows([bad], selection_manifest=manifest, k=2, prompt_messages=prompts, require_complete=False, require_judge=False)


def test_triviaqa_preserves_string_results_and_judge_defines_final_target(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    ids = [row["example_id"] for row in manifest["selected_examples"]]
    prompts = {
        row["example_id"]: [{"role": "user", "content": row["question"]}]
        for row in manifest["selected_examples"]
    }
    rows = [
        sample_row(manifest, example_id, sampled_id, judge=1 - sampled_id)
        for example_id in ids
        for sampled_id in range(2)
    ]
    validated = validate_sample_rows(rows, selection_manifest=manifest, k=2, prompt_messages=prompts, require_complete=True, require_judge=True)
    assert all(row["correctness"] == row["judge_correctness"] for row in validated.values())
    string_gt = aggregate_ground_truth(rows, selection_manifest=manifest, k=2, correctness_field="string_match_correctness")
    judge_gt = aggregate_ground_truth(rows, selection_manifest=manifest, k=2, correctness_field="judge_correctness")
    assert [row["num_correct"] for row in string_gt] == [1, 1]
    assert [row["num_correct"] for row in judge_gt] == [1, 1]
    assert all("exact_rendered_prompt" not in row for row in rows)


def test_non_thinking_preflight_flags_reasoning_and_tags() -> None:
    assert not validate_non_thinking_responses([{"content": "4", "reasoning": None}])
    errors = validate_non_thinking_responses(
        [{"content": "<think>work</think> 4", "reasoning": "work"}]
    )
    assert any("reasoning" in error for error in errors)
    assert any("thinking marker" in error for error in errors)


def test_independent_audit_checks_dual_outputs_and_locked_max_tokens(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    write_immutable_manifest(manifest_path, manifest)
    output = tmp_path / "output"
    output.mkdir()
    (output / "selection_manifest.json").write_text(manifest_path.read_text())
    prompts = {
        row["example_id"]: [{"role": "user", "content": row["question"]}]
        for row in manifest["selected_examples"]
    }
    contract = generation_contract(
        selection_manifest=manifest,
        model_id="model",
        k=2,
        max_tokens=8000,
        temperature=0.7,
        top_p=0.8,
        prompt_messages=prompts,
    )
    state = {
        "sampling_state_version": 1,
        "generation_contract": contract,
        "verification_contracts": {"string_match": {}, "judge": {"judge_model_id": "judge"}},
        "completed_verification_stages": ["string_match", "judge"],
    }
    atomic_write_json(output / "sampling_state.json", state)
    ids = [row["example_id"] for row in manifest["selected_examples"]]
    rows = [
        sample_row(manifest, example_id, sampled_id, judge=1 - sampled_id)
        for example_id in ids
        for sampled_id in range(2)
    ]
    string_rows = []
    for row in rows:
        value = dict(row)
        value["judge_correctness"] = None
        value["correctness"] = value["string_match_correctness"]
        string_rows.append(value)
    atomic_write_jsonl(output / "sampled.jsonl", rows)
    atomic_write_jsonl(output / "sampled_string_match.jsonl", string_rows)
    atomic_write_jsonl(
        output / "ground_truth.jsonl",
        aggregate_ground_truth(rows, selection_manifest=manifest, k=2, correctness_field="judge_correctness"),
    )
    atomic_write_jsonl(
        output / "ground_truth_string_match.jsonl",
        aggregate_ground_truth(string_rows, selection_manifest=manifest, k=2, correctness_field="string_match_correctness"),
    )
    assert audit_test_output(output, manifest_path=manifest_path)["ok"]
    state["generation_contract"]["max_tokens"] = 7999
    atomic_write_json(output / "sampling_state.json", state)
    report = audit_test_output(output, manifest_path=manifest_path)
    assert not report["ok"]
    assert any("max_tokens" in error for error in report["errors"])


def test_launcher_dry_run_has_six_concurrent_worker_chains() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/launch_test_sampling.py", "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["blocking_qwen_preflight"] is True
    assert payload["max_parallel_workers"] == 6
    assert payload["sample_task_concurrency_per_worker"] == 250
    assert payload["max_tokens"] == 8000
    assert len(payload["worker_chains"]) == 6
    assert len(payload["commands"]) == 9
    trivia_workers = [
        worker for worker in payload["worker_chains"] if worker["dataset"] == "triviaqa"
    ]
    mmlu_workers = [
        worker for worker in payload["worker_chains"] if worker["dataset"] == "mmlu"
    ]
    assert len(trivia_workers) == 3
    assert all(len(worker["commands"]) == 2 for worker in trivia_workers)
    assert all("verification_mode=string_match" in worker["commands"][0] for worker in trivia_workers)
    assert all("verification_mode=llm_judge" in worker["commands"][1] for worker in trivia_workers)
    assert len(mmlu_workers) == 3
    assert all(len(worker["commands"]) == 1 for worker in mmlu_workers)
    assert all("max_concurrent=250" in command for command in payload["commands"])
    assert all("inference.max_tokens=8000" in command for command in payload["commands"])
    assert sum("verification_mode=llm_judge" in command for command in payload["commands"]) == 3
    assert all("exact_rendered_prompt" not in command for command in payload["commands"])


def test_launcher_executes_six_first_stages_concurrently_and_chains_judges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = threading.Barrier(6, timeout=5)
    lock = threading.Lock()
    events: list[tuple[str, str]] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check
        worker_id, stage = command
        if stage == "string_match":
            barrier.wait()
        with lock:
            events.append((worker_id, stage))

    monkeypatch.setattr(test_launcher.subprocess, "run", fake_run)
    workers = [
        {
            "worker_id": f"trivia-{index}",
            "commands": [[f"trivia-{index}", "string_match"], [f"trivia-{index}", "judge"]],
        }
        for index in range(3)
    ] + [
        {
            "worker_id": f"mmlu-{index}",
            "commands": [[f"mmlu-{index}", "string_match"]],
        }
        for index in range(3)
    ]
    test_launcher.execute_worker_chains(workers)
    for index in range(3):
        string_event = events.index((f"trivia-{index}", "string_match"))
        judge_event = events.index((f"trivia-{index}", "judge"))
        assert string_event < judge_event
