"""MMLU (Massive Multitask Language Understanding) dataset."""

import re
from typing import Any, Optional

from datasets import load_dataset

from src.data.base import BaseDataset, DataExample


class MMLUDataset(BaseDataset):
    """
    MMLU: Multiple choice questions across 57 subjects.
    
    Questions have 4 choices (A, B, C, D); answer is the correct letter.
    Verification is via string matching on the extracted letter.
    """

    def __init__(
        self,
        split: str = "test",
        max_examples: Optional[int] = None,
        seed: Optional[int] = None,
        hf_path: str = "cais/mmlu",
        hf_name: str = "all",  # Or specific subject like "abstract_algebra"
    ):
        super().__init__(split=split, max_examples=max_examples, seed=seed)
        self.hf_path = hf_path
        self.hf_name = hf_name

    @property
    def name(self) -> str:
        return "mmlu"

    @property
    def verifier_type(self) -> str:
        return "string_match"

    def _load_examples(self) -> list[DataExample]:
        """Load MMLU from HuggingFace."""
        dataset = load_dataset(self.hf_path, self.hf_name, split=self.split)
        
        # Map answer index to letter
        idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
        
        examples = []
        for i, item in enumerate(dataset):
            # Build question with choices
            question = item["question"]
            choices = item["choices"]
            
            formatted_question = f"{question}\n"
            for j, choice in enumerate(choices):
                formatted_question += f"{idx_to_letter[j]}. {choice}\n"
            
            # Get answer letter
            answer = idx_to_letter[item["answer"]]
            
            examples.append(DataExample(
                id=f"mmlu_{self.hf_name}_{self.split}_{i}",
                question=formatted_question.strip(),
                answer=answer,
                metadata={
                    "subject": item.get("subject", self.hf_name),
                    "choices": choices,
                },
            ))
        
        return examples

    def extract_answer(self, response: str) -> Any:
        """
        Extract answer letter from model response.
        
        Looks for:
        1. "The answer is (X)" pattern
        2. Standalone letter at the end
        3. First occurrence of A/B/C/D
        """
        response = response.strip()
        
        # Pattern: "The answer is (X)", "Answer: X", or boxed{X}
        patterns = [
            r"boxed\{([A-D])\}",  # Matches boxed{A}, boxed{B}, boxed{C}, boxed{D}
            r"(?:the\s+)?answer\s+is\s*[\(\[]?([A-D])[\)\]]?",
            r"(?:answer|choice)[\s:]*[\(\[]?([A-D])[\)\]]?",
            r"\(([A-D])\)\s*$",  # Ends with (X)
            r"^([A-D])[\.\)]",  # Starts with X. or X)
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                return matches[-1].upper()
        
        # Fall back to last letter A-D in response
        letters = re.findall(r"\b([A-D])\b", response)
        if letters:
            return letters[-1].upper()
        
        return None

