"""HumanEval code generation dataset."""

import re
from typing import Any, Optional

from datasets import load_dataset

from src.data.base import BaseDataset, DataExample


class HumanEvalDataset(BaseDataset):
    """
    HumanEval: Python function completion benchmark.
    
    Given a function signature and docstring, generate the function body.
    Verification is via executing test cases.
    """

    def __init__(
        self,
        split: str = "test",
        max_examples: Optional[int] = None,
        seed: Optional[int] = None,
        hf_path: str = "openai/openai_humaneval",
    ):
        super().__init__(split=split, max_examples=max_examples, seed=seed)
        self.hf_path = hf_path

    @property
    def name(self) -> str:
        return "humaneval"

    @property
    def verifier_type(self) -> str:
        return "code_execution"

    def _load_examples(self) -> list[DataExample]:
        """Load HumanEval from HuggingFace."""
        dataset = load_dataset(self.hf_path, split=self.split)
        
        examples = []
        for item in dataset:
            task_id = item["task_id"]
            prompt = item["prompt"]
            canonical_solution = item["canonical_solution"]
            test_code = item["test"]
            entry_point = item["entry_point"]
            
            examples.append(DataExample(
                id=task_id,
                question=prompt,
                answer=canonical_solution,
                metadata={
                    "test_code": test_code,
                    "entry_point": entry_point,
                    "prompt": prompt,
                },
            ))
        
        return examples

    def extract_answer(self, response: str) -> Any:
        """
        Extract Python code from model response.
        
        Looks for:
        1. Code in triple backticks
        2. Code after "```python"
        3. Raw code if it looks like Python
        """
        response = response.strip()
        
        # Try to extract from markdown code blocks
        patterns = [
            r"```python\s*(.*?)```",  # ```python ... ```
            r"```\s*(.*?)```",  # ``` ... ```
            r"```(.*?)$",  # Unclosed code block
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                code = match.group(1).strip()
                if code:
                    return code
        
        # If no code blocks, return the whole response
        # (assuming it's raw code)
        return response

    def get_full_code(self, example: DataExample, completion: str) -> str:
        """
        Combine prompt with completion to get full executable code.
        
        Args:
            example: The dataset example with prompt.
            completion: The extracted code completion.
        
        Returns:
            Full Python code ready for execution.
        """
        prompt = example.metadata.get("prompt", example.question)
        
        # If completion already includes the function signature, use as-is
        if "def " in completion:
            return completion
        
        # Otherwise, append completion to prompt
        return prompt + completion

