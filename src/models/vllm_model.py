"""vLLM local inference backend."""

from typing import Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .base import BaseModel, GenerationOutput, HiddenStateMode, UsageStats


class VLLMModel(BaseModel):
    """
    vLLM model implementation for efficient local inference.
    
    Supports:
    - High-throughput batched inference
    - Token logprobs extraction
    - Tensor parallelism for large models
    
    Note: Hidden state extraction is limited in vLLM.
    Use HFModel if you need hidden states for probe training.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        hidden_state_mode: str = "last_token",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        dtype: str = "auto",
    ):
        super().__init__(
            model_id=model_id,
            device=device,
            hidden_state_mode=HiddenStateMode(hidden_state_mode),
        )
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        
        self._llm = LLM(
            model=self.model_id,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get the loaded HuggingFace tokenizer for chat template application."""
        return self._tokenizer

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        """Apply the model's chat template to messages."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @property
    def llm(self) -> LLM:
        """Get the loaded vLLM engine."""
        return self._llm

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
        """Generate text using vLLM."""
        if return_hidden_states:
            raise NotImplementedError(
                "vLLM does not support hidden state extraction. "
                "Use HFModel for hidden state access."
            )

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p,
            stop=stop,
            logprobs=1 if return_logprobs else None,  # Return top-1 logprob
        )

        outputs = self.llm.generate([prompt], sampling_params)
        output = outputs[0]
        
        generated_text = output.outputs[0].text
        
        # Extract tokens and logprobs
        tokens = []
        token_logprobs = []
        
        if return_logprobs and output.outputs[0].logprobs:
            for logprob_dict in output.outputs[0].logprobs:
                # logprob_dict is {token_id: Logprob}
                for token_id, logprob_obj in logprob_dict.items():
                    tokens.append(logprob_obj.decoded_token)
                    token_logprobs.append(logprob_obj.logprob)
                    break  # Only take the first (sampled) token
        
        # Compute usage
        usage = UsageStats(
            prompt_tokens=len(output.prompt_token_ids),
            completion_tokens=len(output.outputs[0].token_ids),
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
        """
        Async generation - falls back to sync for vLLM local.
        
        For async inference, use OpenAICompatModel with a vLLM server.
        """
        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            return_logprobs=return_logprobs,
            return_hidden_states=return_hidden_states,
        )

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
        """Generate from chat messages by applying the model's chat template."""
        prompt = self._apply_chat_template(messages)
        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            return_logprobs=return_logprobs,
            return_hidden_states=return_hidden_states,
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
        """Async version of generate_from_messages() - falls back to sync."""
        return self.generate_from_messages(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            return_logprobs=return_logprobs,
            return_hidden_states=return_hidden_states,
        )

    def get_logprob_for_completion(
        self,
        prompt: str,
        completion: str,
    ) -> float:
        """
        Compute log probability of a specific completion using vLLM.
        
        Uses prompt_logprobs feature to get completion token probabilities.
        """
        # Concatenate prompt and completion
        full_text = prompt + completion
        
        # Use vLLM to compute logprobs for the full sequence
        sampling_params = SamplingParams(
            max_tokens=1,  # Generate minimal tokens
            temperature=0.0,
            prompt_logprobs=0,  # Get logprobs for prompt tokens
        )
        
        outputs = self.llm.generate([full_text], sampling_params)
        output = outputs[0]
        
        # Sum logprobs for completion tokens
        # prompt_logprobs includes all tokens; we need to identify completion portion
        if output.prompt_logprobs is None:
            raise RuntimeError("vLLM did not return prompt logprobs")
        
        # Get number of prompt tokens
        # We need to tokenize to find the boundary
        prompt_tokens = self.llm.get_tokenizer().encode(prompt)
        prompt_len = len(prompt_tokens)
        
        # Sum logprobs for tokens after prompt
        total_logprob = 0.0
        for i, logprob_dict in enumerate(output.prompt_logprobs):
            if i >= prompt_len and logprob_dict is not None:
                # Get the actual token's logprob
                for token_id, logprob_obj in logprob_dict.items():
                    total_logprob += logprob_obj.logprob
                    break
        
        return total_logprob

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
        return_logprobs: bool = False,
        return_hidden_states: bool = False,
    ) -> list[GenerationOutput]:
        """
        Batched generation using vLLM's native batching.
        
        Much more efficient than sequential generation.
        """
        if return_hidden_states:
            raise NotImplementedError(
                "vLLM does not support hidden state extraction. "
                "Use HFModel for hidden state access."
            )

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p,
            stop=stop,
            logprobs=1 if return_logprobs else None,
        )

        outputs = self.llm.generate(prompts, sampling_params)
        
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            
            tokens = []
            token_logprobs = []
            
            if return_logprobs and output.outputs[0].logprobs:
                for logprob_dict in output.outputs[0].logprobs:
                    for token_id, logprob_obj in logprob_dict.items():
                        tokens.append(logprob_obj.decoded_token)
                        token_logprobs.append(logprob_obj.logprob)
                        break
            
            usage = UsageStats(
                prompt_tokens=len(output.prompt_token_ids),
                completion_tokens=len(output.outputs[0].token_ids),
            )

            results.append(GenerationOutput(
                text=generated_text,
                tokens=tokens,
                token_logprobs=token_logprobs,
                hidden_states=None,
                usage=usage,
            ))
        
        return results
