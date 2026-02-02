"""MATH training dataset (excluding MATH-500)."""

from typing import Optional

from datasets import load_dataset

from src.data.base import DataExample
from src.data.datasets.math_base import BaseMathDataset


class MathTrainDataset(BaseMathDataset):
    """
    MATH training set: 11,498 competition math problems.
    
    Derived from the full MATH dataset with MATH-500 problems excluded.
    Questions are competition-level math problems from various domains.
    Answers are mathematical expressions (can include fractions, variables, etc.).
    Verification via math_verify library or LLM judge.
    """

    def __init__(
        self,
        split: str = "train",
        max_examples: Optional[int] = None,
        seed: Optional[int] = None,
        hf_path: str = "appier-ai-research/MATH-train-11498",
        verification_mode: str = "math_verify",
        llm_judge_model: Optional[str] = "openai/gpt-oss-20b",
        llm_judge_base_url: Optional[str] = None,
        llm_judge_max_concurrent: int = 1000,
    ):
        super().__init__(
            split=split,
            max_examples=max_examples,
            seed=seed,
            hf_path=hf_path,
            verification_mode=verification_mode,
            llm_judge_model=llm_judge_model,
            llm_judge_base_url=llm_judge_base_url,
            llm_judge_max_concurrent=llm_judge_max_concurrent,
        )

    @property
    def name(self) -> str:
        return "math-train"

    def _load_examples(self) -> list[DataExample]:
        """Load MATH training set from HuggingFace."""
        dataset = load_dataset(self.hf_path, split=self.split)
        
        examples = []
        for i, item in enumerate(dataset):
            examples.append(DataExample(
                id=f"math_train_{self.split}_{i}",
                question=item["problem"],
                answer=item["answer"],
                metadata={
                    "subject": item.get("subject", ""),
                    "level": item.get("level", ""),
                },
            ))
        
        return examples
