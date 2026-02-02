"""Base class for MATH competition datasets with shared verification logic."""

import asyncio
import re
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from src.data.base import BaseDataset, DataExample

if TYPE_CHECKING:
    from src.models.openai_compat import OpenAICompatModel


def get_llm_judge_prompt(ground_truth: str, predicted_answer: str) -> str:
    prompt = f"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale in your final answer.

    Expression 1: {ground_truth}
    Expression 2: {predicted_answer}
    """.strip()
    return prompt


class BaseMathDataset(BaseDataset):
    """
    Base class for MATH competition datasets.
    
    Provides shared functionality:
    - Answer extraction from \\boxed{} format
    - Verification via math_verify library or LLM judge
    
    Subclasses must implement:
    - name property
    - _load_examples method
    """

    def __init__(
        self,
        split: str = "test",
        max_examples: Optional[int] = None,
        seed: Optional[int] = None,
        hf_path: str = "",
        verification_mode: str = "math_verify",  # or "llm_judge"
        llm_judge_model: Optional[str] = "openai/gpt-oss-20b",
        llm_judge_base_url: Optional[str] = None,
        llm_judge_max_concurrent: int = 1000,
    ):
        super().__init__(split=split, max_examples=max_examples, seed=seed)
        self.hf_path = hf_path
        self.verification_mode = verification_mode
        self.llm_judge_model_id = llm_judge_model
        self.llm_judge_base_url = llm_judge_base_url
        self.llm_judge_max_concurrent = llm_judge_max_concurrent
        self._llm_judge_model = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name identifier."""
        pass

    @property
    def verifier_type(self) -> str:
        return self.verification_mode

    @property
    def llm_judge_model(self) -> "OpenAICompatModel":
        """Lazy-load the LLM judge model."""
        if self._llm_judge_model is None:
            from src.models.openai_compat import OpenAICompatModel
            self._llm_judge_model = OpenAICompatModel(
                model_id=self.llm_judge_model_id,
                base_url=self.llm_judge_base_url,
            )
        return self._llm_judge_model

    def _extract_boxed_content(self, text: str) -> Optional[str]:
        """
        Extract content from the last \\boxed{...} handling nested braces.
        
        Returns the content inside the last \\boxed{} or None if not found.
        """
        # Find all \boxed{ occurrences
        pattern = r"\\boxed\{"
        matches = list(re.finditer(pattern, text))
        
        if not matches:
            return None
        
        # Get the last occurrence
        last_match = matches[-1]
        start_idx = last_match.end()  # Position right after \boxed{
        
        # Count braces to find matching closing brace
        brace_count = 1
        idx = start_idx
        
        while idx < len(text) and brace_count > 0:
            if text[idx] == "{":
                brace_count += 1
            elif text[idx] == "}":
                brace_count -= 1
            idx += 1
        
        if brace_count == 0:
            # Extract content (excluding the final closing brace)
            return text[start_idx:idx - 1]
        
        return None

    def extract_answer(self, response: str) -> Any:
        """
        Extract mathematical answer from model response.
        
        Looks for the last \\boxed{X} pattern, handling nested braces.
        """
        extracted = self._extract_boxed_content(response)
        if extracted is not None:
            return extracted.strip()
        return None

    async def verify_answer_by_math_verify(self, extracted: Any, ground_truth: DataExample) -> bool:
        """Use math_verify library to check mathematical equivalence."""
        if extracted is None:
            return False
        
        def _verify():
            try:
                from math_verify import parse, verify
                
                gold_parsed = parse(ground_truth.answer)
                extracted_parsed = parse(str(extracted))
                
                return verify(gold_parsed, extracted_parsed)
            except Exception:
                # If parsing fails, fall back to string comparison
                return str(extracted).strip() == str(ground_truth.answer).strip()
        
        # Run synchronous math_verify in thread pool for async compatibility
        return await asyncio.to_thread(_verify)

    async def verify_answer_by_llm_judge(self, extracted: Any, ground_truth: DataExample) -> bool:
        """Use LLM judge to verify if extracted answer is mathematically equivalent."""
        if extracted is None:
            return False
        
        prompt = get_llm_judge_prompt(
            ground_truth=ground_truth.answer,
            predicted_answer=str(extracted),
        )
        
        messages = [{"role": "user", "content": prompt}]
        output = await self.llm_judge_model.generate_from_messages_async(
            messages=messages,
            max_tokens=2048,
            temperature=1.0,
            top_p=1.0,
        )
        # Parse "Yes" or "No" - only "Yes" is True
        response_text = output.text.strip().lower()
        return response_text == "yes" or response_text.endswith("yes")

    async def verify_answer(self, extracted: Any, ground_truth: DataExample) -> bool:
        """
        Verify if extracted answer is mathematically equivalent to ground truth.
        
        Uses either math_verify library or LLM judge based on verification_mode.
        """
        return await getattr(self, f"verify_answer_by_{self.verification_mode}")(extracted, ground_truth)
