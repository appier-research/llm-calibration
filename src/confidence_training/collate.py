"""Batch collation for feature and prompt confidence datasets."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from .data import ConfidenceExample, render_prompts, tokenize_rendered_prompts


def collate_linear(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "inputs": {"features": torch.stack([row["features"] for row in rows])},
        "targets": torch.stack([row["target"] for row in rows]).float(),
        "example_ids": [str(row["example_id"]) for row in rows],
    }


class PromptCollator:
    def __init__(
        self,
        tokenizer: Any,
        *,
        device: torch.device | str,
        max_sequence_length: int | None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.chat_template_kwargs = dict(chat_template_kwargs or {})

    def __call__(self, rows: Sequence[ConfidenceExample]) -> dict[str, Any]:
        rendered = render_prompts(
            self.tokenizer,
            rows,
            chat_template_kwargs=self.chat_template_kwargs,
        )
        inputs = tokenize_rendered_prompts(
            self.tokenizer,
            rendered,
            device=self.device,
            max_sequence_length=self.max_sequence_length,
        )
        return {
            "inputs": inputs,
            "targets": torch.tensor(
                [row.ground_truth.expected_accuracy for row in rows],
                dtype=torch.float32,
                device=self.device,
            ),
            "example_ids": [row.example_id for row in rows],
        }

