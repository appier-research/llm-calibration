"""Base verifier interface for checking answer correctness."""

from abc import ABC, abstractmethod
from typing import Any

from src.data.base import DataExample


class BaseVerifier(ABC):
    """
    Abstract base class for answer verification.
    
    Verifiers determine whether a model's response is correct
    given the ground truth answer.
    
    Different verification strategies:
    - StringMatch: Exact or normalized string comparison
    - CodeExecution: Run code and check output
    - LLMJudge: Use another LLM to assess correctness
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Verifier type identifier."""
        pass

    @abstractmethod
    def verify(
        self,
        example: DataExample,
        response: Any,
    ) -> bool:
        """
        Check if the response is correct for the given example.
        
        Args:
            example: The dataset example with ground truth.
            response: The extracted answer from model output
                     (already parsed by dataset.extract_answer).
        
        Returns:
            True if the response is correct, False otherwise.
        """
        pass

    def verify_batch(
        self,
        examples: list[DataExample],
        responses: list[Any],
    ) -> list[bool]:
        """
        Verify multiple responses.
        
        Default implementation is sequential; subclasses (like LLMJudge)
        can override for batched verification.
        """
        return [
            self.verify(ex, resp)
            for ex, resp in zip(examples, responses)
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

