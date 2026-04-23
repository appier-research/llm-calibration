"""AIME (test): combined AIME 2024 + 2025 + 2026 = 90 problems."""

from typing import Any, Optional

from datasets import load_dataset
from dotenv import load_dotenv

from src.data.base import DataExample
from src.data.datasets.aime25 import AIME25Dataset

# Load HF_TOKEN etc. from .env if present (harmless if file is absent).
load_dotenv()


class AIMETestDataset(AIME25Dataset):
    """
    AIME test split: union of three HuggingFace AIME repos (30 problems each,
    N=90 total):

    * ``HuggingFaceH4/aime_2024`` (split=train) — AIME 2024 I & II
    * ``math-ai/aime25`` (split=test) — AIME 2025 I & II
    * ``MathArena/aime_2026`` (split=train) — AIME 2026 I & II

    The three repos have slightly different schemas, so each source specifies
    the HF ``split`` plus the column names for question / answer / id. Answer
    extraction (``\\boxed{...}``) and ``string_match`` verification are
    inherited from :class:`AIME25Dataset`.
    """

    # Default set of sources. Can be overridden in the Hydra config.
    DEFAULT_SOURCES: list[dict] = [
        {
            "hf_path": "HuggingFaceH4/aime_2024",
            "split": "train",
            "year": 2024,
            "question_field": "problem",
            "answer_field": "answer",
            "id_field": "id",
        },
        {
            "hf_path": "math-ai/aime25",
            "split": "test",
            "year": 2025,
            "question_field": "problem",
            "answer_field": "answer",
            "id_field": "id",
        },
        {
            "hf_path": "MathArena/aime_2026",
            "split": "train",
            "year": 2026,
            "question_field": "problem",
            "answer_field": "answer",
            "id_field": "problem_idx",
        },
    ]

    def __init__(
        self,
        split: str = "test",  # logical split name; per-source HF split lives in ``sources``
        max_examples: Optional[int] = None,
        seed: Optional[int] = None,
        sources: Optional[list[dict]] = None,
    ):
        # ``hf_path`` is unused for this class (we load multiple repos);
        # pass a placeholder to satisfy the parent signature.
        super().__init__(
            split=split,
            max_examples=max_examples,
            seed=seed,
            hf_path="",
        )
        self.sources = sources if sources is not None else self.DEFAULT_SOURCES

    @property
    def name(self) -> str:
        return "aime-test"

    def _load_examples(self) -> list[DataExample]:
        """Load and concatenate problems from all configured AIME sources."""
        examples: list[DataExample] = []
        for src in self.sources:
            dataset = load_dataset(src["hf_path"], split=src["split"])
            year = int(src["year"])
            q_field = src["question_field"]
            a_field = src["answer_field"]
            id_field = src["id_field"]

            for item in dataset:
                raw_id = item[id_field]
                # ID format ``aime_test_<year>_<original-id>`` — guaranteed
                # unique across sources via the year prefix.
                examples.append(DataExample(
                    id=f"aime_test_{year}_{raw_id}",
                    question=item[q_field],
                    answer=str(item[a_field]),
                    metadata={
                        "year": year,
                        "source_hf_path": src["hf_path"],
                        "source_split": src["split"],
                        "source_id": raw_id,
                    },
                ))

        return examples

    def extract_answer(self, response: str) -> Any:
        """Inherited from :class:`AIME25Dataset` — extracts the last ``\\boxed{<digits>}``."""
        return super().extract_answer(response)
