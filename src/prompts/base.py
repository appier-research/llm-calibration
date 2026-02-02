"""Base prompt builder interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PromptConfig:
    """Configuration for a prompt template."""
    system: str = ""
    user_template: str = ""
    format_instructions: dict[str, str] | None = None
    few_shot_examples: list[dict[str, str]] | None = None
    num_few_shot: int = 0


class PromptBuilder(ABC):
    """
    Abstract base class for prompt construction.
    
    Separates prompt engineering from datasets and estimators.
    Prompts can be swapped independently via config.
    """

    def __init__(self, config: PromptConfig):
        self.config = config

    @abstractmethod
    def build(self, **kwargs) -> str:
        """
        Build a complete prompt from the template and inputs.
        
        Args:
            **kwargs: Template variables (question, answer, etc.)
        
        Returns:
            Formatted prompt string.
        """
        pass

    def format_template(self, template: str, **kwargs) -> str:
        """Format a template string with provided variables."""
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")

    def build_chat_messages(self, **kwargs) -> list[dict[str, str]]:
        """
        Build chat-format messages for chat models.
        
        Returns:
            List of {"role": ..., "content": ...} dicts.
        """
        messages = []
        
        if self.config.system:
            messages.append({
                "role": "system",
                "content": self.config.system.strip(),
            })
        
        # Add few-shot examples if configured
        if self.config.few_shot_examples and self.config.num_few_shot > 0:
            for example in self.config.few_shot_examples[:self.config.num_few_shot]:
                if "user" in example:
                    messages.append({"role": "user", "content": example["user"]})
                if "assistant" in example:
                    messages.append({"role": "assistant", "content": example["assistant"]})
        
        # Add the main user message
        user_content = self.build_user_content(**kwargs)
        messages.append({
            "role": "user",
            "content": user_content,
        })
        
        return messages

    @abstractmethod
    def build_user_content(self, **kwargs) -> str:
        """Build the user message content."""
        pass

