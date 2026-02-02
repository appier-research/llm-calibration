"""Base dataset interface for calibration experiments."""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

@dataclass
class DataExample:
    """
    A single example from a dataset.
    
    Attributes:
        id: Unique identifier for this example.
        question: The question/prompt to answer.
        answer: Ground truth answer (format depends on dataset).
        metadata: Additional dataset-specific fields.
    """
    id: str
    question: str
    answer: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDataset(ABC):
    """
    Abstract base class for datasets.
    
    Datasets are responsible for:
    - Loading and iterating over examples
    - Defining what verifier type they use
    - Extracting answers from model responses (dataset-specific parsing)
    
    Note: Prompt construction is NOT the dataset's job - that's handled
    by the prompt layer.
    """

    def __init__(
        self,
        split: str = "test",
        max_examples: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            split: Dataset split to use (train/val/test).
            max_examples: Limit number of examples (for debugging).
            seed: Random seed for shuffling/sampling.
        """
        self.split = split
        self.max_examples = max_examples
        self.seed = seed
        self._examples: Optional[list[DataExample]] = None
        if seed is not None:
            print(f"Setting random seed to {seed}")
            random.seed(seed)
        else:
            print("No random seed provided")

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name identifier."""
        pass

    @property
    @abstractmethod
    def verifier_type(self) -> str:
        """
        The type of verifier this dataset requires.
        
        Returns one of: 'string_match', 'code_execution', 'llm_judge'
        """
        pass

    @abstractmethod
    def _load_examples(self) -> list[DataExample]:
        """
        Load all examples from the data source.
        
        Subclasses implement this to load from HuggingFace, local files, etc.
        """
        pass

    @abstractmethod
    def extract_answer(self, response: str) -> Any:
        """
        Extract the answer from a model response.
        
        Each dataset defines its own extraction logic since answer formats
        vary (e.g., GSM8K expects a number, MMLU expects A/B/C/D).
        
        Args:
            response: Raw model output text.
        
        Returns:
            Extracted answer in dataset-appropriate format.
        """
        pass

    async def verify_answer(self, extracted: Any, ground_truth: "DataExample") -> bool | None:
        """
        Optional dataset-specific verification.
        
        Override this method when the dataset needs custom verification logic
        (e.g., TriviaQA checking against multiple aliases).
        
        Args:
            extracted: The extracted answer from model response.
            ground_truth: The dataset example with ground truth.
        
        Returns:
            True/False if dataset handles verification, None to use external verifier.
        """
        return None

    def load(self) -> None:
        """Load examples into memory."""
        if self._examples is None:
            self._examples = self._load_examples()
            if self.max_examples is not None:
                print(f"Limiting dataset to {self.max_examples} randomly selected examples")
                self._examples = random.sample(self._examples, self.max_examples)

    @property
    def examples(self) -> list[DataExample]:
        """Get loaded examples, loading if necessary."""
        if self._examples is None:
            self.load()
        return self._examples  # type: ignore

    def __iter__(self) -> Iterator[DataExample]:
        """Iterate over dataset examples."""
        return iter(self.examples)

    def __len__(self) -> int:
        """Number of examples in the dataset."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> DataExample:
        """Get example by index."""
        return self.examples[idx]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(split={self.split!r}, n={len(self) if self._examples else '?'})"

