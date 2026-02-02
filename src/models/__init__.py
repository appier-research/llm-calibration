from .base import BaseModel, GenerationOutput, HiddenStateMode, UsageStats
from .hf_model import HFModel
from .openai_compat import OpenAICompatModel

__all__ = [
    "BaseModel",
    "GenerationOutput",
    "HiddenStateMode",
    "UsageStats",
    "HFModel",
    "OpenAICompatModel",
]

try:
    from .vllm_model import VLLMModel
    __all__.append("VLLMModel")
except ImportError:
    pass
