# LLM Calibration Research Codebase
"""
rethinking-calibration: A research codebase for LLM calibration.

Main modules:
- data: Datasets and data loading
- models: LLM inference backends (vLLM, HuggingFace, OpenAI-compatible)
- prompts: Prompt construction for predictions and confidence
- estimators: Confidence estimation methods (logprob, verbalized, etc.)
- verifiers: Answer verification (string match, code execution, LLM judge)
- metrics: Calibration and discrimination metrics
- probes: Learned confidence probes (linear, MLP)
- experiment: Experiment running and tracking
- inference: Sampling, caching, and async utilities
"""

__version__ = "0.1.0"
