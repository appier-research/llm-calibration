import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from src.confidence_training.data import (
    audit_dataset_folder,
    canonical_question,
    find_split_overlaps,
    last_non_padding_indices,
    load_confidence_examples,
    load_ground_truth,
    render_prompts,
    tokenize_rendered_prompts,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_folder(root: Path) -> Path:
    root.mkdir()
    ground_truth = [
        {
            "example_id": "a",
            "question": "  What   is A? ",
            "num_samples": 10,
            "num_correct": 3,
            "expected_accuracy": 0.3,
        },
        {
            "example_id": "b",
            "question": "What is B?",
            "num_samples": 4,
            "num_correct": 4,
            "expected_accuracy": 1.0,
        },
    ]
    write_jsonl(root / "ground_truth.jsonl", ground_truth)
    write_jsonl(
        root / "sampled.jsonl",
        [
            {"example_id": "b", "sampled_id": 0, "prompt": [{"role": "user", "content": "B"}]},
            {"example_id": "a", "sampled_id": 0, "prompt": [{"role": "user", "content": "A"}]},
            {"example_id": "a", "sampled_id": 1, "prompt": [{"role": "user", "content": "ignored"}]},
        ],
    )
    probing = root / "probing_dataset_cc"
    probing.mkdir()
    save_file({"hidden_states": torch.ones(2, 3, 4)}, probing / "hidden_states.safetensors")
    save_file({"targets": torch.tensor([0.3, 1.0])}, probing / "targets.safetensors")
    (probing / "metadata.json").write_text(
        json.dumps({"n_examples": 2, "n_layers": 3, "hidden_dim": 4, "example_ids": ["a", "b"]})
    )
    return root


def test_ground_truth_and_prompt_join_are_strict(tmp_path: Path) -> None:
    folder = make_folder(tmp_path / "data")
    examples = load_confidence_examples(folder)
    assert [row.example_id for row in examples] == ["a", "b"]
    assert examples[0].prompt[0]["content"] == "A"
    report = audit_dataset_folder(folder)
    assert report["ok"]


def test_ground_truth_rejects_duplicates_and_bad_counts(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    row = {
        "example_id": "a",
        "question": "A",
        "num_samples": 2,
        "num_correct": 3,
        "expected_accuracy": 1.5,
    }
    write_jsonl(path, [row])
    with pytest.raises(ValueError):
        load_ground_truth(path)
    row.update(num_correct=1, expected_accuracy=0.5)
    write_jsonl(path, [row, row])
    with pytest.raises(ValueError, match="Duplicate"):
        load_ground_truth(path)


def test_audit_detects_metadata_and_target_mismatch(tmp_path: Path) -> None:
    folder = make_folder(tmp_path / "data")
    probing = folder / "probing_dataset_cc"
    save_file({"targets": torch.tensor([1.0, 0.3])}, probing / "targets.safetensors")
    report = audit_dataset_folder(folder)
    assert not report["ok"]
    assert any("targets.safetensors" in message for message in report["errors"])


def test_last_non_padding_supports_left_and_right_padding() -> None:
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0]])
    assert last_non_padding_indices(mask).tolist() == [1, 3, 2]
    with pytest.raises(ValueError):
        last_non_padding_indices(torch.zeros(1, 2, dtype=torch.long))


class FakeTokenizer:
    model_max_length = 3

    def __call__(self, rows, **kwargs):
        del rows, kwargs
        return {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }


def test_context_overflow_fails_without_truncation() -> None:
    with pytest.raises(ValueError, match="truncation is disabled"):
        tokenize_rendered_prompts(
            FakeTokenizer(), ["prompt"], device="cpu", max_sequence_length=None
        )


class DateTemplateTokenizer:
    chat_template = '{{ strftime_now("%Y-%m-%d") }}'

    def apply_chat_template(self, rows, **kwargs):
        del rows
        return [kwargs["chat_template"]]


def test_prompt_rendering_can_freeze_historical_template_date() -> None:
    tokenizer = DateTemplateTokenizer()
    example = type("Example", (), {"prompt": [{"role": "user", "content": "Q"}]})()
    rendered = render_prompts(
        tokenizer,
        [example],
        chat_template_kwargs={"template_date": "2026-04-21"},
    )
    assert rendered == ['{{ "2026-04-21" }}']
    with pytest.raises(ValueError, match="does not use strftime_now"):
        tokenizer.chat_template = "{{ messages }}"
        render_prompts(
            tokenizer,
            [example],
            chat_template_kwargs={"template_date": "2026-04-21"},
        )


def test_normalized_exact_question_overlap(tmp_path: Path) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    base = {"num_samples": 1, "num_correct": 1, "expected_accuracy": 1.0}
    write_jsonl(first_path, [{**base, "example_id": "a", "question": "What  is A?"}])
    write_jsonl(second_path, [{**base, "example_id": "b", "question": " What is A? "}])
    overlap = find_split_overlaps(load_ground_truth(first_path), load_ground_truth(second_path))
    assert overlap["questions"] == [canonical_question("What is A?")]
