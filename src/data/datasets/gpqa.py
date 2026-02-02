"""GPQA (Graduate-Level Google-Proof Q&A) dataset."""

import random
import re
from typing import Any, Optional

from datasets import load_dataset

from src.data.base import BaseDataset, DataExample


class GPQADataset(BaseDataset):
    """
    GPQA: Expert-level multiple choice questions in biology, physics, and chemistry.
    
    Questions have 4 choices (A, B, C, D); answer is the correct letter.
    Choices are shuffled since the correct answer is always first in the raw data.
    Verification is via string matching on the extracted letter.
    """

    def __init__(
        self,
        split: str = "train",  # GPQA only has train split
        max_examples: Optional[int] = None,
        seed: Optional[int] = None,
        hf_path: str = "Idavidrein/gpqa",
        hf_name: str = "gpqa_main",  # gpqa_main, gpqa_diamond, gpqa_extended
    ):
        super().__init__(split=split, max_examples=max_examples, seed=seed)
        self.hf_path = hf_path
        self.hf_name = hf_name

    @property
    def name(self) -> str:
        return "gpqa"

    @property
    def verifier_type(self) -> str:
        return "string_match"

    def _load_examples(self) -> list[DataExample]:
        """Load GPQA from HuggingFace."""
        dataset = load_dataset(self.hf_path, self.hf_name, split=self.split)
        
        idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
        
        examples = []
        for i, item in enumerate(dataset):
            # Gather all choices (correct answer + 3 incorrect)
            choices = [
                item["Correct Answer"],
                item["Incorrect Answer 1"],
                item["Incorrect Answer 2"],
                item["Incorrect Answer 3"],
            ]
            
            # Shuffle choices deterministically using seed + example index
            rng = random.Random((self.seed or 0) + i)
            correct_idx = 0  # Correct answer is at index 0 before shuffle
            
            # Create indices and shuffle them
            indices = list(range(4))
            rng.shuffle(indices)
            
            # Reorder choices and find new position of correct answer
            shuffled_choices = [choices[idx] for idx in indices]
            new_correct_idx = indices.index(correct_idx)
            answer = idx_to_letter[new_correct_idx]
            
            # Build formatted question with choices
            question = item["Question"]
            formatted_question = f"{question}\n"
            for j, choice in enumerate(shuffled_choices):
                formatted_question += f"{idx_to_letter[j]}. {choice}\n"
            
            examples.append(DataExample(
                id=f"gpqa_{self.hf_name}_{self.split}_{i}",
                question=formatted_question.strip(),
                answer=answer,
                metadata={
                    "subdomain": item.get("Subdomain", ""),
                    "explanation": item.get("Explanation", ""),
                    "choices": shuffled_choices,
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
            r"boxed\{([A-D])\}",
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

