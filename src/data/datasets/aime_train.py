"""AIME (train): gneubig/aime-1983-2024 excluding year 2024 (N=919)."""

from typing import Any, Optional

from datasets import load_dataset
from dotenv import load_dotenv

from src.data.base import DataExample
from src.data.datasets.aime25 import AIME25Dataset

# Load HF_TOKEN etc. from .env if present (harmless if file is absent).
load_dotenv()


class AIMETrainDataset(AIME25Dataset):
    """
    AIME train split: problems from ``gneubig/aime-1983-2024`` excluding
    year 2024, giving N=919 problems (out of 933 total; 14 from 2024 dropped).

    Year 2024 is excluded so it does not overlap with AIME-2024-based eval
    splits used elsewhere. Answer extraction (``\\boxed{...}``) and
    ``string_match`` verification are inherited from :class:`AIME25Dataset`.
    """

    def __init__(
        self,
        split: str = "train",  # gneubig/aime-1983-2024 only has a train split
        max_examples: Optional[int] = None,
        seed: Optional[int] = None,
        hf_path: str = "gneubig/aime-1983-2024",
        exclude_year: Optional[int] = 2024,
    ):
        super().__init__(
            split=split,
            max_examples=max_examples,
            seed=seed,
            hf_path=hf_path,
        )
        self.exclude_year = exclude_year

    @property
    def name(self) -> str:
        return "aime-train"

    def _load_examples(self) -> list[DataExample]:
        """Load gneubig/aime-1983-2024 and drop rows whose ``Year`` equals ``exclude_year``."""
        dataset = load_dataset(self.hf_path, split=self.split)

        examples: list[DataExample] = []
        for item in dataset:
            if self.exclude_year is not None and int(item["Year"]) == self.exclude_year:
                continue

            # ``ID`` in this HF repo is stable and human-readable, e.g. "1983-1".
            examples.append(DataExample(
                id=f"aime_train_{item['ID']}",
                question=item["Question"],
                answer=str(item["Answer"]),
                metadata={
                    "year": int(item["Year"]),
                    "problem_number": int(item["Problem Number"]),
                    "part": item.get("Part"),
                },
            ))

        return examples

    def extract_answer(self, response: str) -> Any:
        """Inherited from :class:`AIME25Dataset` — extracts the last ``\\boxed{<digits>}``."""
        return super().extract_answer(response)
