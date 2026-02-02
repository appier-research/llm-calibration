"""HuggingFace Transformers model with hidden state extraction."""

import asyncio
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseModel, GenerationOutput, HiddenStateMode, UsageStats


class HFModel(BaseModel):
    """
    HuggingFace Transformers model with full control over inference.
    
    Supports:
    - Hidden state extraction (required for probe training)
    - Quantization (8-bit, 4-bit)
    - Various attention implementations
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        hidden_state_mode: str = "last_token",
        dtype: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__(
            model_id=model_id,
            device=device,
            hidden_state_mode=HiddenStateMode(hidden_state_mode),
        )
        self.dtype = dtype
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.attn_implementation = attn_implementation
        
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model, self._tokenizer = self._load_model()

    def _resolve_dtype(self) -> Optional[torch.dtype]:
        """Convert string dtype to torch.dtype."""
        if self.dtype == "auto":
            return None  # Let transformers decide
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype)

    def _resolve_device(self) -> str:
        """Resolve device string to actual device."""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def _load_model(self) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model and tokenizer."""
        dtype = self._resolve_dtype()

        model_kwargs = {
            "trust_remote_code": True,
            "output_hidden_states": True,  # Always enable for hidden state extraction
        }
        
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        
        if self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        elif self.device != "auto":
            model_kwargs["device_map"] = self._resolve_device()
        else:
            model_kwargs["device_map"] = "auto"
        
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs,
        )
        self._model.eval()
        
        return self._model, self._tokenizer

    def _extract_hidden_states(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract hidden states according to configured mode.
        
        Args:
            hidden_states: Tuple of hidden states from all layers.
                          Shape: (num_layers, batch, seq_len, hidden_dim)
            attention_mask: Attention mask. Shape: (batch, seq_len)
        
        Returns:
            Extracted hidden states. Shape depends on mode:
            - last_token: (batch, hidden_dim)
            - mean_pool: (batch, hidden_dim)
            - all_tokens: (batch, seq_len, hidden_dim)
        """
        # Use last layer hidden states
        last_hidden = hidden_states[-1]  # (batch, seq_len, hidden_dim)
        
        if self.hidden_state_mode == HiddenStateMode.LAST_TOKEN:
            # Get last non-padding token for each sequence
            seq_lens = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
            return last_hidden[batch_indices, seq_lens]  # (batch, hidden_dim)
        
        elif self.hidden_state_mode == HiddenStateMode.MEAN_POOL:
            # Mean pool over non-padding tokens
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            summed = (last_hidden * mask).sum(dim=1)  # (batch, hidden_dim)
            lengths = mask.sum(dim=1)  # (batch, 1)
            return summed / lengths  # (batch, hidden_dim)
        
        elif self.hidden_state_mode == HiddenStateMode.ALL_TOKENS:
            return last_hidden  # (batch, seq_len, hidden_dim)
        
        else:
            raise ValueError(f"Unknown hidden state mode: {self.hidden_state_mode}")

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
        model, tokenizer = self._model, self._tokenizer
        device = next(model.parameters()).device
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]
        
        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_tokens - prompt_len,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "output_scores": return_logprobs,
            "return_dict_in_generate": True,
            "output_hidden_states": return_hidden_states,
        }
        
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        
        # Handle stop sequences via stopping criteria
        if stop:
            from transformers import StoppingCriteria, StoppingCriteriaList
            
            class StopSequenceCriteria(StoppingCriteria):
                def __init__(self, stop_seqs: list[str], tokenizer):
                    self.stop_seqs = stop_seqs
                    self.tokenizer = tokenizer
                
                def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
                    generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    return any(seq in generated_text for seq in self.stop_seqs)
            
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
                StopSequenceCriteria(stop, tokenizer)
            ])
        
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, **gen_kwargs)
        
        generated_ids = outputs.sequences[0][prompt_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Truncate at stop sequence if present
        if stop:
            for seq in stop:
                if seq in text:
                    text = text[:text.index(seq)]
                    break
        
        tokens: list[str] = []
        token_logprobs: list[float] = []
        
        if return_logprobs and outputs.scores:
            for i, score in enumerate(outputs.scores):
                if i >= len(generated_ids):
                    break
                token_id = generated_ids[i].item()
                log_probs = torch.log_softmax(score[0], dim=-1)
                token_logprobs.append(log_probs[token_id].item())
                tokens.append(tokenizer.decode([token_id]))
        
        hidden_states_tensor: Optional[torch.Tensor] = None
        if return_hidden_states and outputs.hidden_states:
            # outputs.hidden_states is tuple of (num_generated_tokens, tuple of layer tensors)
            # Each layer tensor: (batch, 1, hidden_dim) for generated tokens
            # We want hidden states at the last generated position
            last_step_hidden = outputs.hidden_states[-1]  # Last generated token's hidden states
            # Stack all layers: (num_layers, batch, 1, hidden_dim)
            all_layers = torch.stack(last_step_hidden, dim=0)
            # Take last layer, squeeze: (batch, hidden_dim)
            hidden_states_tensor = all_layers[-1].squeeze(1).cpu()
        
        usage = UsageStats(
            prompt_tokens=prompt_len,
            completion_tokens=len(generated_ids),
        )
        
        return GenerationOutput(
            text=text,
            tokens=tokens,
            token_logprobs=token_logprobs,
            hidden_states=hidden_states_tensor,
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
        # HF models are synchronous; run in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                return_logprobs=return_logprobs,
                return_hidden_states=return_hidden_states,
            ),
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
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
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
        """Async version of generate_from_messages()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_from_messages(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                return_logprobs=return_logprobs,
                return_hidden_states=return_hidden_states,
            ),
        )

    def get_logprob_for_completion(
        self,
        prompt: str,
        completion: str,
    ) -> float:
        """
        Compute log probability of completion given prompt.
        
        Concatenates prompt + completion and computes log probs
        for the completion tokens only.
        """
        model, tokenizer = self._model, self._tokenizer
        device = next(model.parameters()).device
        
        # Tokenize separately to find boundary
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        
        # Full sequence
        full_ids = tokenizer.encode(prompt + completion, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(full_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)
        
        # Compute log probs for each position
        log_probs = torch.log_softmax(logits[0], dim=-1)  # (seq_len, vocab_size)
        
        # Sum log probs for completion tokens
        # Position i predicts token i+1, so we start from len(prompt_ids)-1
        prompt_len = len(prompt_ids)
        total_logprob = 0.0
        
        for i in range(prompt_len, full_ids.shape[1]):
            token_id = full_ids[0, i].item()
            # Log prob comes from position i-1 predicting position i
            total_logprob += log_probs[i - 1, token_id].item()
        
        return total_logprob

    def get_hidden_states(
        self,
        text: str,
    ) -> torch.Tensor:
        """
        Get hidden states for input text without generation.
        
        Useful for probe training where we just need representations.
        
        Args:
            text: Input text to encode.
        
        Returns:
            Hidden states according to configured mode.
        """
        model, tokenizer = self._model, self._tokenizer
        device = next(model.parameters()).device
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        return self._extract_hidden_states(
            outputs.hidden_states,
            inputs.attention_mask,
        ).cpu()

    def get_hidden_states_batch(
        self,
        texts: list[str],
    ) -> torch.Tensor:
        """
        Get hidden states for a batch of texts.
        
        Args:
            texts: List of input texts.
        
        Returns:
            Hidden states. Shape: (batch, hidden_dim) for last_token/mean_pool,
            or (batch, max_seq_len, hidden_dim) for all_tokens.
        """
        model, tokenizer = self._model, self._tokenizer
        device = next(model.parameters()).device
        
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        return self._extract_hidden_states(
            outputs.hidden_states,
            inputs.attention_mask,
        ).cpu()
