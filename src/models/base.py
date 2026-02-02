"""Base model interface for LLM inference."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch


class HiddenStateMode(str, Enum):
    """How to extract hidden states from the model."""
    LAST_TOKEN = "last_token"
    MEAN_POOL = "mean_pool"
    ALL_TOKENS = "all_tokens"


@dataclass
class UsageStats:
    """Token usage statistics for a generation."""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "UsageStats") -> "UsageStats":
        return UsageStats(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


@dataclass
class GenerationOutput:
    """Output from a model generation call."""
    text: str
    tokens: list[str] = field(default_factory=list)
    token_logprobs: list[float] = field(default_factory=list)
    hidden_states: Optional[torch.Tensor] = None
    usage: UsageStats = field(default_factory=UsageStats)


class BaseModel(ABC):
    """
    Abstract base class for LLM inference.
    
    Provides unified interface for:
    - Text generation with optional logprobs
    - Hidden state extraction
    - Token probability computation
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        hidden_state_mode: HiddenStateMode = HiddenStateMode.LAST_TOKEN,
    ):
        self.model_id = model_id
        self.device = device
        self.hidden_state_mode = hidden_state_mode

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
        return_logprobs: bool = False,
        return_hidden_states: bool = False,
    ) -> GenerationOutput:
        """
        Generate text completion for the given prompt.
        
        Args:
            prompt: Input text to complete.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling parameter.
            stop: Stop sequences.
            return_logprobs: Whether to return per-token log probabilities.
            return_hidden_states: Whether to return hidden states.
        
        Returns:
            GenerationOutput with text, tokens, logprobs, hidden_states, and usage.
        """
        pass

    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
        return_logprobs: bool = False,
        return_hidden_states: bool = False,
    ) -> GenerationOutput:
        """Async version of generate() for concurrent API calls."""
        pass

    @abstractmethod
    def get_logprob_for_completion(
        self,
        prompt: str,
        completion: str,
    ) -> float:
        """
        Compute the log probability of a specific completion given a prompt.
        
        Used for P(True) style confidence estimation where we need
        P(completion | prompt) for a fixed completion string.
        
        Args:
            prompt: The conditioning prompt.
            completion: The completion to score.
        
        Returns:
            Log probability of the completion.
        """
        pass

    @abstractmethod
    def generate_from_messages(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
        return_logprobs: bool = False,
        return_hidden_states: bool = False,
    ) -> GenerationOutput:
        """
        Generate text from chat messages, applying the model's chat template.
        
        Args:
            messages: List of {"role": str, "content": str} dicts.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling parameter.
            stop: Stop sequences.
            return_logprobs: Whether to return per-token log probabilities.
            return_hidden_states: Whether to return hidden states.
        
        Returns:
            GenerationOutput with text, tokens, logprobs, hidden_states, and usage.
        """
        pass

    @abstractmethod
    async def generate_from_messages_async(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
        return_logprobs: bool = False,
        return_hidden_states: bool = False,
    ) -> GenerationOutput:
        """Async version of generate_from_messages() for concurrent API calls."""
        pass

    def generate_batch(
        self,
        prompts: list[str],
        **kwargs,
    ) -> list[GenerationOutput]:
        """
        Generate completions for multiple prompts.
        
        Default implementation is sequential; subclasses can override
        for batched inference.
        """
        return [self.generate(p, **kwargs) for p in prompts]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id={self.model_id!r})"

