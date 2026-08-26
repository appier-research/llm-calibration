"""Strict data loading, prompt rendering, hashing, and audit utilities."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset


AUDIT_SCOPES = ("probing_artifacts", "adapted_lm")


@dataclass(frozen=True, slots=True)
class GroundTruthRow:
    example_id: str
    question: str
    num_samples: int
    num_correct: int
    expected_accuracy: float


@dataclass(frozen=True, slots=True)
class ConfidenceExample:
    ground_truth: GroundTruthRow
    prompt: list[dict[str, str]]

    @property
    def example_id(self) -> str:
        return self.ground_truth.example_id


def canonical_question(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).strip().split())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return sha256_bytes(encoded)


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield line_number, value


def load_ground_truth(path: str | Path) -> list[GroundTruthRow]:
    source = Path(path)
    rows: list[GroundTruthRow] = []
    seen: set[str] = set()
    for line_number, value in _iter_jsonl(source):
        example_id = str(value.get("example_id", ""))
        if not example_id:
            raise ValueError(f"{source}:{line_number} is missing example_id")
        if example_id in seen:
            raise ValueError(f"Duplicate example_id in {source}: {example_id}")
        seen.add(example_id)
        question_value = value.get("question", value.get("query"))
        if question_value is None:
            raise ValueError(f"{source}:{line_number} is missing question/query")
        num_samples = int(value["num_samples"])
        num_correct = int(value["num_correct"])
        expected_accuracy = float(value["expected_accuracy"])
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive for {example_id}")
        if not 0 <= num_correct <= num_samples:
            raise ValueError(f"Invalid num_correct for {example_id}")
        if not math.isfinite(expected_accuracy) or not 0.0 <= expected_accuracy <= 1.0:
            raise ValueError(f"Invalid expected_accuracy for {example_id}")
        exact = num_correct / num_samples
        if not math.isclose(expected_accuracy, exact, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"expected_accuracy mismatch for {example_id}: "
                f"{expected_accuracy} != {num_correct}/{num_samples}"
            )
        rows.append(
            GroundTruthRow(
                example_id=example_id,
                question=str(question_value),
                num_samples=num_samples,
                num_correct=num_correct,
                expected_accuracy=expected_accuracy,
            )
        )
    if not rows:
        raise ValueError(f"Ground truth is empty: {source}")
    return rows


def load_first_prompts(
    path: str | Path,
    required_ids: set[str] | None = None,
) -> dict[str, list[dict[str, str]]]:
    source = Path(path)
    prompts: dict[str, list[dict[str, str]]] = {}
    for line_number, value in _iter_jsonl(source):
        example_id = str(value.get("example_id", ""))
        if required_ids is not None and example_id not in required_ids:
            continue
        if example_id in prompts:
            continue
        prompt = value.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"{source}:{line_number} has an invalid prompt")
        normalized: list[dict[str, str]] = []
        for message in prompt:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise ValueError(f"{source}:{line_number} has an invalid prompt message")
            normalized.append(
                {"role": str(message["role"]), "content": str(message["content"])}
            )
        prompts[example_id] = normalized
        if required_ids is not None and len(prompts) == len(required_ids):
            break
    return prompts


def load_confidence_examples(folder: str | Path) -> list[ConfidenceExample]:
    root = Path(folder)
    ground_truth = load_ground_truth(root / "ground_truth.jsonl")
    required = {row.example_id for row in ground_truth}
    prompts = load_first_prompts(root / "sampled.jsonl", required)
    missing = sorted(required - set(prompts))
    if missing:
        raise ValueError(f"Missing prompts in {root / 'sampled.jsonl'}: {missing[:10]}")
    return [ConfidenceExample(row, prompts[row.example_id]) for row in ground_truth]


def ground_truth_hash(rows: Sequence[GroundTruthRow]) -> str:
    return hash_json([asdict(row) for row in rows])


def prompt_hash(examples: Sequence[ConfidenceExample]) -> str:
    return hash_json(
        [{"example_id": row.example_id, "prompt": row.prompt} for row in examples]
    )


def rendered_prompt_hash(example_ids: Sequence[str], rendered: Sequence[str]) -> str:
    return hash_json(list(zip(example_ids, rendered, strict=True)))


def last_non_padding_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be two-dimensional")
    if torch.any(attention_mask.sum(dim=1) <= 0):
        raise ValueError("Every sequence must contain a non-padding token")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    return positions.expand_as(attention_mask).masked_fill(attention_mask == 0, -1).max(1).values


def render_prompts(
    tokenizer: Any,
    examples: Sequence[ConfidenceExample],
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    kwargs = dict(chat_template_kwargs or {})
    template_date = kwargs.pop("template_date", None)
    if template_date is not None:
        template = kwargs.pop("chat_template", getattr(tokenizer, "chat_template", None))
        if not template:
            raise ValueError("template_date requires a tokenizer chat_template")
        replacements = (
            'strftime_now("%Y-%m-%d")',
            "strftime_now('%Y-%m-%d')",
        )
        for expression in replacements:
            if expression in template:
                template = template.replace(expression, json.dumps(str(template_date)))
                break
        else:
            raise ValueError(
                "template_date was supplied, but the chat template does not use "
                "strftime_now for a YYYY-MM-DD date"
            )
        kwargs["chat_template"] = template
    return list(
        tokenizer.apply_chat_template(
            [example.prompt for example in examples],
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
    )


def tokenize_rendered_prompts(
    tokenizer: Any,
    rendered: Sequence[str],
    *,
    device: torch.device | str,
    max_sequence_length: int | None,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(rendered),
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    lengths = encoded["attention_mask"].sum(dim=1)
    configured_limit = max_sequence_length
    model_limit = getattr(tokenizer, "model_max_length", None)
    finite_model_limit = (
        int(model_limit)
        if isinstance(model_limit, int) and model_limit < 10**9
        else None
    )
    limit = configured_limit if configured_limit is not None else finite_model_limit
    if limit is not None and int(lengths.max()) > limit:
        raise ValueError(
            f"Prompt length {int(lengths.max())} exceeds configured/model limit {limit}; "
            "truncation is disabled for this experiment."
        )
    return {key: value.to(device) for key, value in encoded.items()}


class LinearFeatureDataset(Dataset):
    """Mean-across-layer feature dataset aligned to ground-truth order."""

    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        example_ids: Sequence[str],
    ) -> None:
        if features.ndim != 2 or targets.ndim != 1:
            raise ValueError("features must be 2-D and targets must be 1-D")
        if len(features) != len(targets) or len(features) != len(example_ids):
            raise ValueError("Feature dataset lengths do not match")
        self.features = features.float()
        self.targets = targets.float()
        self.example_ids = list(example_ids)

    @classmethod
    def load(cls, folder: str | Path, probing_dataset_name: str) -> "LinearFeatureDataset":
        root = Path(folder)
        probing = root / probing_dataset_name
        metadata = json.loads((probing / "metadata.json").read_text())
        hidden = load_file(probing / "hidden_states.safetensors")["hidden_states"]
        targets = load_file(probing / "targets.safetensors")["targets"]
        if hidden.ndim != 3:
            raise ValueError(f"Expected hidden states [N,L,D], got {tuple(hidden.shape)}")
        return cls(hidden.mean(dim=1), targets, metadata["example_ids"])

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "features": self.features[index],
            "target": self.targets[index],
            "example_id": self.example_ids[index],
        }


class PromptConfidenceDataset(Dataset):
    def __init__(self, examples: Sequence[ConfidenceExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> ConfidenceExample:
        return self.examples[index]


def audit_dataset_folder(
    folder: str | Path,
    *,
    scope: str = "probing_artifacts",
    probing_dataset_name: str = "probing_dataset_cc",
    expected_count: int | None = None,
) -> dict[str, Any]:
    if scope not in AUDIT_SCOPES:
        raise ValueError(f"Unknown audit scope: {scope}")
    root = Path(folder)
    errors: list[str] = []
    try:
        rows = load_ground_truth(root / "ground_truth.jsonl")
        examples = load_confidence_examples(root)
    except Exception as error:  # audit reports every folder rather than aborting matrix scans
        return {"folder": str(root), "ok": False, "errors": [str(error)]}
    if expected_count is not None and len(rows) != expected_count:
        errors.append(
            f"ground_truth.jsonl count {len(rows)} does not match expected_count={expected_count}"
        )
    probing = root / probing_dataset_name
    if scope == "probing_artifacts":
        try:
            metadata = json.loads((probing / "metadata.json").read_text())
            targets = load_file(probing / "targets.safetensors")["targets"].float()
            hidden = load_file(probing / "hidden_states.safetensors")["hidden_states"]
            expected_ids = [row.example_id for row in rows]
            expected_targets = torch.tensor(
                [row.expected_accuracy for row in rows], dtype=torch.float32
            )
            if metadata.get("n_examples") != len(rows):
                errors.append("metadata n_examples does not match ground_truth.jsonl")
            if metadata.get("example_ids") != expected_ids:
                errors.append("metadata example_ids do not match ground-truth order")
            if targets.shape != expected_targets.shape or not torch.equal(
                targets, expected_targets
            ):
                errors.append(
                    "targets.safetensors does not exactly match ground-truth targets"
                )
            if hidden.ndim != 3 or hidden.shape[0] != len(rows):
                errors.append("hidden_states.safetensors shape does not match ground truth")
        except Exception as error:
            errors.append(str(error))
    report = {
        "folder": str(root),
        "audit_scope": scope,
        "ok": not errors,
        "errors": errors,
        "n_examples": len(rows),
        "example_ids": [row.example_id for row in rows],
        "questions": [row.question for row in rows],
        "normalized_questions": [canonical_question(row.question) for row in rows],
        "ground_truth_hash": ground_truth_hash(rows),
        "prompt_hash": prompt_hash(examples),
        "ordered_data_hash": ground_truth_hash(rows),
        "ordered_prompt_hash": prompt_hash(examples),
    }
    if scope == "probing_artifacts":
        report["rebuild_command"] = (
            "uv run python scripts/construct_probing_dataset.py "
            f"--model-id <MODEL_ID> --ground-truth-path {root / 'ground_truth.jsonl'} "
            f"--sampled-path {root / 'sampled.jsonl'} --output-dir {probing}"
        )
    return report


def find_split_overlaps(
    first: Sequence[GroundTruthRow], second: Sequence[GroundTruthRow]
) -> dict[str, list[str]]:
    first_ids = {row.example_id for row in first}
    second_ids = {row.example_id for row in second}
    first_questions = {canonical_question(row.question) for row in first}
    second_questions = {canonical_question(row.question) for row in second}
    return {
        "example_ids": sorted(first_ids & second_ids),
        "questions": sorted(first_questions & second_questions),
    }
