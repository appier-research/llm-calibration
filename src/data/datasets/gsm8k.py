"""GSM8K math word problems dataset."""

import re
from typing import Any, Optional

from datasets import load_dataset

from src.data.base import BaseDataset, DataExample


class GSM8KDataset(BaseDataset):
    """
    GSM8K: Grade School Math 8K benchmark.
    
    Questions are math word problems; answers are numeric.
    Verification is via string matching on extracted numbers.
    """

    def __init__(
        self,
        split: str = "test",
        max_examples: Optional[int] = None,
        seed: Optional[int] = None,
        hf_path: str = "openai/gsm8k",
        hf_name: str = "main",
    ):
        super().__init__(split=split, max_examples=max_examples, seed=seed)
        self.hf_path = hf_path
        self.hf_name = hf_name

    @property
    def name(self) -> str:
        return "gsm8k"

    @property
    def verifier_type(self) -> str:
        return "string_match"

    def _load_examples(self) -> list[DataExample]:
        """Load GSM8K from HuggingFace."""
        dataset = load_dataset(self.hf_path, self.hf_name, split=self.split)
        
        examples = []
        for i, item in enumerate(dataset):
            # Extract the final answer from the solution
            # GSM8K format: "...#### 42"
            answer = self._extract_answer_from_solution(item["answer"])
            
            examples.append(DataExample(
                id=f"gsm8k_{self.split}_{i}",
                question=item["question"],
                answer=answer,
                metadata={
                    "full_solution": item["answer"],
                },
            ))
        
        return examples

    def _extract_answer_from_solution(self, solution: str) -> str:
        """Extract the final numeric answer from GSM8K solution format."""
        # GSM8K answers end with "#### <number>"
        match = re.search(r"####\s*([^\n]+)", solution)
        if match:
            return match.group(1).strip()
        return solution.strip()

    def extract_answer(self, response: str) -> Any:
        """
        Extract numeric answer from model response.
        
        Looks for: \boxed{X} format (following our designed prompt)
        """
        # First, try to find \boxed{X} format
        matches = re.findall(r"\\boxed\{([0-9,.\-]+)[^}]*\}", response)
        if matches:
            return self._normalize_number(matches[-1])
        return None

    def _normalize_number(self, num_str: str) -> float | None:
        """Normalize a number string to float."""
        try:
            # Remove commas
            cleaned = num_str.replace(",", "")
            return float(cleaned)
        except ValueError:
            return None
