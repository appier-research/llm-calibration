"""OpenAI-compatible API client for vLLM/SGLang servers."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import asyncio
from typing import Optional

from openai import AsyncOpenAI, OpenAI
from transformers import AutoTokenizer

from .base import BaseModel, GenerationOutput, HiddenStateMode, UsageStats


class OpenAICompatModel(BaseModel):
    """
    OpenAI-compatible API client for remote inference.
    
    Works with:
    - vLLM server (python -m vllm.entrypoints.openai.api_server)
    - SGLang server
    - Any OpenAI-compatible endpoint
    
    Supports:
    - Async generation with semaphore-based rate limiting
    - Token logprobs extraction
    - Chat template application via HuggingFace tokenizer
    
    Note: Hidden states are not available via API.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        hidden_state_mode: str = "last_token",
        timeout: int = 1800,
        max_retries: int = 3,
        device: Optional[str] = None,  # Not used, but kept for interface compatibility
        tokenizer_id: Optional[str] = None,  # Defaults to model_id if not specified
    ):
        super().__init__(
            model_id=model_id,
            device=device,
            hidden_state_mode=HiddenStateMode(hidden_state_mode),
        )
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.tokenizer_id = tokenizer_id or model_id
        
        self._client = None
        self._async_client = None
        self._tokenizer = None

    @property
    def client(self) -> OpenAI:
        """Get sync client."""
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    @property
    def async_client(self) -> AsyncOpenAI:
        """Get async client."""
        if self._async_client is None:
            import httpx
            self._async_client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=1000,
                        max_keepalive_connections=100,
                    )
                )
            )
        return self._async_client

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Lazy-load HuggingFace tokenizer for chat template application."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_id,
                trust_remote_code=True,
            )
        return self._tokenizer

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        """Apply the model's chat template to messages."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

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
        """Generate text using OpenAI-compatible API."""
        if return_hidden_states:
            raise NotImplementedError(
                "Hidden states are not available via API. "
                "Use HFModel for hidden state access."
            )

        # Use completions API for raw text generation
        response = self.client.completions.create(
            model=self.model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            logprobs=1 if return_logprobs else None,
        )

        choice = response.choices[0]
        generated_text = choice.text

        # Extract tokens and logprobs
        tokens = []
        token_logprobs = []
        
        if return_logprobs and choice.logprobs:
            tokens = choice.logprobs.tokens or []
            token_logprobs = choice.logprobs.token_logprobs or []

        # Compute usage
        usage = UsageStats(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )

        return GenerationOutput(
            text=generated_text,
            tokens=tokens,
            token_logprobs=token_logprobs,
            hidden_states=None,
            usage=usage,
        )

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
        """Async generation for concurrent API calls."""
        if return_hidden_states:
            raise NotImplementedError(
                "Hidden states are not available via API. "
                "Use HFModel for hidden state access."
            )

        response = await self.async_client.completions.create(
            model=self.model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            logprobs=1 if return_logprobs else None,
        )

        choice = response.choices[0]
        generated_text = choice.text

        tokens = []
        token_logprobs = []
        
        if return_logprobs and choice.logprobs:
            tokens = choice.logprobs.tokens or []
            token_logprobs = choice.logprobs.token_logprobs or []

        usage = UsageStats(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )

        return GenerationOutput(
            text=generated_text,
            tokens=tokens,
            token_logprobs=token_logprobs,
            hidden_states=None,
            usage=usage,
        )

    def messages_to_prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        """Convert chat messages to prompt tokens."""
        return len(self.tokenizer.encode(self._apply_chat_template(messages)))

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
        """Generate from chat messages using chat completions API."""
        if return_hidden_states:
            raise NotImplementedError(
                "Hidden states are not available via API. "
                "Use HFModel for hidden state access."
            )

        prompt_tokens = self.messages_to_prompt_tokens(messages)
        max_completion_tokens = max_tokens - prompt_tokens

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            logprobs=return_logprobs,
        )

        choice = response.choices[0]
        generated_text = choice.message.content or ""

        tokens = []
        token_logprobs = []

        if return_logprobs and choice.logprobs and choice.logprobs.content:
            tokens = [t.token for t in choice.logprobs.content]
            token_logprobs = [t.logprob for t in choice.logprobs.content]

        usage = UsageStats(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )

        return GenerationOutput(
            text=generated_text,
            tokens=tokens,
            token_logprobs=token_logprobs,
            hidden_states=None,
            usage=usage,
        )

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
        """Async version of generate_from_messages()."""
        if return_hidden_states:
            raise NotImplementedError(
                "Hidden states are not available via API. "
                "Use HFModel for hidden state access."
            )

        prompt_tokens = self.messages_to_prompt_tokens(messages)
        max_completion_tokens = max_tokens - prompt_tokens

        response = await self.async_client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            logprobs=return_logprobs,
        )

        choice = response.choices[0]
        generated_text = choice.message.content or ""
        if choice.message.reasoning is not None:
            generated_text = choice.message.reasoning + "</think>" + generated_text

        tokens = []
        token_logprobs = []

        if return_logprobs and choice.logprobs and choice.logprobs.content:
            tokens = [t.token for t in choice.logprobs.content]
            token_logprobs = [t.logprob for t in choice.logprobs.content]

        usage = UsageStats(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )

        return GenerationOutput(
            text=generated_text,
            tokens=tokens,
            token_logprobs=token_logprobs,
            hidden_states=None,
            usage=usage,
        )

    def get_logprob_for_completion(
        self,
        prompt: str,
        completion: str,
    ) -> float:
        """
        Compute log probability of a specific completion.
        
        Uses echo=True to get logprobs for the prompt+completion.
        """
        # Concatenate and request logprobs with echo
        full_text = prompt + completion
        
        response = self.client.completions.create(
            model=self.model_id,
            prompt=full_text,
            max_tokens=0,  # No new generation
            echo=True,  # Return prompt tokens with logprobs
            logprobs=1,
        )

        choice = response.choices[0]
        
        if not choice.logprobs or not choice.logprobs.token_logprobs:
            raise RuntimeError("Server did not return logprobs")
        
        # Need to find where completion starts
        # Tokenize prompt to find boundary
        # This is approximate - for exact results, use HFModel
        prompt_response = self.client.completions.create(
            model=self.model_id,
            prompt=prompt,
            max_tokens=0,
            echo=True,
            logprobs=1,
        )
        prompt_len = len(prompt_response.choices[0].logprobs.tokens or [])
        
        # Sum logprobs for completion tokens
        all_logprobs = choice.logprobs.token_logprobs
        completion_logprobs = all_logprobs[prompt_len:]
        
        # Filter None values and sum
        total_logprob = sum(lp for lp in completion_logprobs if lp is not None)
        
        return total_logprob

    async def generate_batch_async(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
        return_logprobs: bool = False,
        max_concurrent: int = 16,
    ) -> list[GenerationOutput]:
        """
        Async batched generation with semaphore-based rate limiting.
        
        Args:
            prompts: List of prompts to generate.
            max_concurrent: Maximum concurrent API calls.
            Other args: Same as generate().
        
        Returns:
            List of GenerationOutputs in same order as prompts.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_semaphore(prompt: str) -> GenerationOutput:
            async with semaphore:
                return await self.generate_async(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    return_logprobs=return_logprobs,
                )

        tasks = [generate_with_semaphore(p) for p in prompts]
        return await asyncio.gather(*tasks)
