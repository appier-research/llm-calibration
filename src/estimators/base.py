"""Base estimator interface for confidence estimation."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.models.base import BaseModel


class BaseEstimator(ABC):
    """
    Abstract base class for confidence estimation.
    
    Two types of confidence estimation:
    
    1. estimate_v(model, x, y): "Verification confidence"
       Given a question x and proposed answer y, how confident is the model
       that y is correct? Used for scoring existing responses.
    
    2. estimate_c(model, x): "Correctness confidence"  
       How confident is the model that it CAN answer x correctly?
       This generates a response internally and returns confidence.
    
    Estimation methods vary:
    - TokenLogprob: Sum/mean of token log probabilities
    - P(True): Probability of "True" given "Is this correct?"
    - Verbalized: Ask model to state confidence as 0-100%
    - Consistency: Agreement across multiple samples
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Estimator type identifier."""
        pass

    @property
    def requires_logprobs(self) -> bool:
        """Whether this estimator needs token log probabilities."""
        return False

    @property
    def requires_hidden_states(self) -> bool:
        """Whether this estimator needs hidden state access."""
        return False

    @abstractmethod
    def estimate_v(
        self,
        model: "BaseModel",
        question: str,
        answer: Any,
        **kwargs,
    ) -> float:
        """
        Estimate confidence that the given answer is correct.
        
        Args:
            model: The language model to use.
            question: The question/prompt.
            answer: A proposed answer to evaluate.
            **kwargs: Estimator-specific parameters.
        
        Returns:
            Confidence score in [0, 1].
        """
        pass

    @abstractmethod
    def estimate_c(
        self,
        model: "BaseModel",
        question: str,
        **kwargs,
    ) -> tuple[Any, float]:
        """
        Generate an answer and estimate confidence in correctness.
        
        Args:
            model: The language model to use.
            question: The question/prompt.
            **kwargs: Estimator-specific parameters.
        
        Returns:
            Tuple of (generated_answer, confidence_score).
            Confidence score is in [0, 1].
        """
        pass

    async def estimate_v_async(
        self,
        model: "BaseModel",
        question: str,
        answer: Any,
        **kwargs,
    ) -> float:
        """Async version of estimate_v for concurrent estimation."""
        # Default: fall back to sync version
        return self.estimate_v(model, question, answer, **kwargs)

    async def estimate_c_async(
        self,
        model: "BaseModel",
        question: str,
        **kwargs,
    ) -> tuple[Any, float]:
        """Async version of estimate_c for concurrent estimation."""
        # Default: fall back to sync version
        return self.estimate_c(model, question, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

