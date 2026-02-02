"""Prediction prompt builder for generating answers to questions."""

from typing import Optional
from omegaconf import DictConfig

from .base import PromptBuilder, PromptConfig


class PredictionPromptBuilder(PromptBuilder):
    """
    Builds prompts for generating answers to questions.
    
    Supports:
    - System prompts
    - User templates with {question} placeholder
    - Dataset-specific format instructions
    - Few-shot examples
    """

    def __init__(
        self,
        system: str = "",
        user_template: str = "{question}",
        format_instructions: Optional[dict[str, str]] = None,
        few_shot_examples: Optional[list[dict[str, str]]] = None,
        num_few_shot: int = 0,
    ):
        config = PromptConfig(
            system=system,
            user_template=user_template,
            format_instructions=format_instructions or {},
            few_shot_examples=few_shot_examples,
            num_few_shot=num_few_shot,
        )
        super().__init__(config)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "PredictionPromptBuilder":
        """Create from Hydra config."""
        return cls(
            system=cfg.get("system", ""),
            user_template=cfg.get("user_template", "{question}"),
            format_instructions=dict(cfg.get("format_instructions", {})),
            few_shot_examples=list(cfg.get("few_shot_examples", [])) if cfg.get("few_shot_examples") else None,
            num_few_shot=cfg.get("num_few_shot", 0),
        )

    def build(
        self,
        question: str,
        dataset_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Build a complete prediction prompt.
        
        Args:
            question: The question to answer.
            dataset_name: Optional dataset name for format instructions.
            **kwargs: Additional template variables.
        
        Returns:
            Formatted prompt string.
        """
        parts = []
        
        # System prompt
        if self.config.system:
            parts.append(self.config.system.strip())
            parts.append("")  # Blank line
        
        # Few-shot examples
        if self.config.few_shot_examples and self.config.num_few_shot > 0:
            for example in self.config.few_shot_examples[:self.config.num_few_shot]:
                if "user" in example:
                    parts.append(f"Q: {example['user']}")
                if "assistant" in example:
                    parts.append(f"A: {example['assistant']}")
                parts.append("")
        
        # User template with question
        user_content = self.build_user_content(
            question=question,
            dataset_name=dataset_name,
            **kwargs,
        )
        parts.append(user_content)
        
        return "\n".join(parts)

    def build_user_content(
        self,
        question: str,
        dataset_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Build the user message content."""
        # Format the user template
        content = self.format_template(
            self.config.user_template,
            question=question,
            **kwargs,
        )
        
        # Append dataset-specific format instructions
        if dataset_name and self.config.format_instructions:
            format_inst = self.config.format_instructions.get(dataset_name)
            if format_inst:
                content = f"{content.strip()}\n\n{format_inst}"
        
        return content.strip()

    def build_for_completion(
        self,
        question: str,
        dataset_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Build prompt for completion-style models (non-chat).
        
        Same as build() but explicitly for completion APIs.
        """
        return self.build(question=question, dataset_name=dataset_name, **kwargs)


