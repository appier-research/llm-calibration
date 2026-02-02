"""AIME 2025 math competition dataset."""

import re
from typing import Any, Optional

from datasets import load_dataset

from src.data.base import BaseDataset, DataExample


class AIME25Dataset(BaseDataset):
    """
    AIME 2025: American Invitational Mathematics Examination 2025.
    
    Questions are challenging math competition problems.
    Answers are integers from 000 to 999.
    Verification is via string matching on extracted numbers.
    """

    def __init__(
        self,
        split: str = "test",
        max_examples: Optional[int] = None,
        seed: Optional[int] = None,
        hf_path: str = "math-ai/aime25",
    ):
        super().__init__(split=split, max_examples=max_examples, seed=seed)
        self.hf_path = hf_path

    @property
    def name(self) -> str:
        return "aime25"

    @property
    def verifier_type(self) -> str:
        return "string_match"

    def _load_examples(self) -> list[DataExample]:
        """Load AIME25 from HuggingFace."""
        dataset = load_dataset(self.hf_path, split=self.split)
        
        examples = []
        for item in dataset:
            examples.append(DataExample(
                id=f"aime25_{self.split}_{item['id']}",
                question=item["problem"],
                answer=str(item["answer"]),  # Keep as string for matching
                metadata={},
            ))
        
        return examples

    def extract_answer(self, response: str) -> Any:
        """
        Extract integer answer from model response.
        
        AIME answers are always integers 000-999.
        
        Looks for: \boxed{X} format
        """
        matches = re.findall(r"\\boxed\{(\d+)\}", response)
        if matches:
            return matches[-1].lstrip("0") or "0"
        return None
