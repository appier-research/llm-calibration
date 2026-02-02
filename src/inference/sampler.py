"""Sampler for generating multiple responses per question."""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional, Union

from src.models.base import BaseModel, GenerationOutput, UsageStats
from .cache import ResponseCache
from .semaphore import AsyncSemaphore


@dataclass
class SamplingResult:
    """Result of sampling k responses for a single question."""
    question_id: str
    prompt: str  # Either raw prompt or JSON-serialized messages for cache key
    responses: list[GenerationOutput]
    total_usage: UsageStats = field(default_factory=UsageStats)

    @property
    def num_samples(self) -> int:
        return len(self.responses)

    @property
    def texts(self) -> list[str]:
        """Get just the response texts."""
        return [r.text for r in self.responses]


class Sampler:
    """
    Samples multiple responses per question for ground truth computation.
    
    Supports:
    - Configurable temperature, top_p, top_k
    - Caching for resumption
    - Async batched generation for API-based models
    """

    def __init__(
        self,
        model: BaseModel,
        cache: Optional[ResponseCache] = None,
        semaphore: Optional[AsyncSemaphore] = None,
    ):
        self.model = model
        self.cache = cache
        self.semaphore = semaphore or AsyncSemaphore(max_concurrent=16)

    def sample(
        self,
        k: int,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
        return_logprobs: bool = False,
        question_id: Optional[str] = None,
        prompt: Optional[str] = None,
        messages: Optional[list[dict[str, str]]] = None,
    ) -> SamplingResult:
        """
        Sample k responses for a single prompt or messages (sync).
        
        Args:
            k: Number of samples to generate.
            max_tokens: Maximum tokens per response.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            stop: Stop sequences.
            return_logprobs: Whether to return logprobs.
            question_id: Optional ID for tracking.
            prompt: Raw prompt string (mutually exclusive with messages).
            messages: Chat messages list (mutually exclusive with prompt).
        
        Returns:
            SamplingResult with k responses.
        """
        if prompt is None and messages is None:
            raise ValueError("Either prompt or messages must be provided")
        if prompt is not None and messages is not None:
            raise ValueError("Cannot provide both prompt and messages")
        
        # Use messages JSON as cache key if using messages
        cache_key = prompt if prompt is not None else json.dumps(messages, sort_keys=True)
        
        responses = []
        total_usage = UsageStats()
        
        params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
            "return_logprobs": return_logprobs,
        }
        
        for i in range(k):
            # Create unique cache key per sample
            sample_params = {**params, "sample_idx": i}
            
            # Check cache
            cached = None
            if self.cache:
                cached = self.cache.get(cache_key, self.model.model_id, sample_params)
            
            if cached is not None:
                output = cached
            else:
                if messages is not None:
                    output = self.model.generate_from_messages(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                        return_logprobs=return_logprobs,
                    )
                else:
                    output = self.model.generate(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                        return_logprobs=return_logprobs,
                    )
                # Cache the result
                if self.cache:
                    self.cache.put(cache_key, self.model.model_id, sample_params, output)
            
            responses.append(output)
            total_usage = total_usage + output.usage
        
        return SamplingResult(
            question_id=question_id or "",
            prompt=cache_key,
            responses=responses,
            total_usage=total_usage,
        )

    async def sample_async(
        self,
        k: int,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
        return_logprobs: bool = False,
        question_id: Optional[str] = None,
        prompt: Optional[str] = None,
        messages: Optional[list[dict[str, str]]] = None,
    ) -> SamplingResult:
        """
        Sample k responses for a single prompt or messages (async).
        
        Uses semaphore for rate limiting when making concurrent API calls.
        
        Args:
            k: Number of samples to generate.
            max_tokens: Maximum tokens per response.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            stop: Stop sequences.
            return_logprobs: Whether to return logprobs.
            question_id: Optional ID for tracking.
            prompt: Raw prompt string (mutually exclusive with messages).
            messages: Chat messages list (mutually exclusive with prompt).
        """
        if prompt is None and messages is None:
            raise ValueError("Either prompt or messages must be provided")
        if prompt is not None and messages is not None:
            raise ValueError("Cannot provide both prompt and messages")
        
        # Use messages JSON as cache key if using messages
        cache_key = prompt if prompt is not None else json.dumps(messages, sort_keys=True)
        
        params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
            "return_logprobs": return_logprobs,
        }
        
        async def sample_one(idx: int) -> GenerationOutput:
            sample_params = {**params, "sample_idx": idx}
            
            # Check cache
            if self.cache:
                cached = self.cache.get(cache_key, self.model.model_id, sample_params)
                if cached is not None:
                    return cached
            
            async with self.semaphore.acquire():
                if messages is not None:
                    output = await self.model.generate_from_messages_async(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                        return_logprobs=return_logprobs,
                    )
                else:
                    output = await self.model.generate_async(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                        return_logprobs=return_logprobs,
                    )
            
            if self.cache:
                self.cache.put(cache_key, self.model.model_id, sample_params, output)
            
            return output
        
        tasks = [sample_one(i) for i in range(k)]
        responses = await asyncio.gather(*tasks)
        
        total_usage = UsageStats()
        for r in responses:
            total_usage = total_usage + r.usage
        
        return SamplingResult(
            question_id=question_id or "",
            prompt=cache_key,
            responses=list(responses),
            total_usage=total_usage,
        )

    def sample_batch(
        self,
        k: int,
        prompts: Optional[list[str]] = None,
        messages_list: Optional[list[list[dict[str, str]]]] = None,
        question_ids: Optional[list[str]] = None,
        **kwargs,
    ) -> list[SamplingResult]:
        """
        Sample k responses for multiple prompts or messages (sync).
        
        Args:
            k: Number of samples per item.
            prompts: List of raw prompts (mutually exclusive with messages_list).
            messages_list: List of message lists (mutually exclusive with prompts).
            question_ids: Optional IDs for tracking.
        """
        if prompts is None and messages_list is None:
            raise ValueError("Either prompts or messages_list must be provided")
        if prompts is not None and messages_list is not None:
            raise ValueError("Cannot provide both prompts and messages_list")
        
        items = prompts if prompts is not None else messages_list
        if question_ids is None:
            question_ids = [str(i) for i in range(len(items))]
        
        results = []
        for i, (item, qid) in enumerate(zip(items, question_ids)):
            if prompts is not None:
                result = self.sample(k=k, prompt=item, question_id=qid, **kwargs)
            else:
                result = self.sample(k=k, messages=item, question_id=qid, **kwargs)
            results.append(result)
        
        return results

    async def sample_batch_async(
        self,
        k: int,
        prompts: Optional[list[str]] = None,
        messages_list: Optional[list[list[dict[str, str]]]] = None,
        question_ids: Optional[list[str]] = None,
        **kwargs,
    ) -> list[SamplingResult]:
        """
        Sample k responses for multiple prompts or messages (async).
        
        Runs all sampling concurrently with semaphore-based rate limiting.
        
        Args:
            k: Number of samples per item.
            prompts: List of raw prompts (mutually exclusive with messages_list).
            messages_list: List of message lists (mutually exclusive with prompts).
            question_ids: Optional IDs for tracking.
        """
        if prompts is None and messages_list is None:
            raise ValueError("Either prompts or messages_list must be provided")
        if prompts is not None and messages_list is not None:
            raise ValueError("Cannot provide both prompts and messages_list")
        
        items = prompts if prompts is not None else messages_list
        if question_ids is None:
            question_ids = [str(i) for i in range(len(items))]
        
        if prompts is not None:
            tasks = [
                self.sample_async(k=k, prompt=item, question_id=qid, **kwargs)
                for item, qid in zip(items, question_ids)
            ]
        else:
            tasks = [
                self.sample_async(k=k, messages=item, question_id=qid, **kwargs)
                for item, qid in zip(items, question_ids)
            ]
        
        return await asyncio.gather(*tasks)


