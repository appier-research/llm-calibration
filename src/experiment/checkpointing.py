"""Checkpointing for experiment resumption."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Optional


@dataclass
class ExampleResult:
    """Result for a single example."""
    example_id: str
    question: str
    ground_truth: Any
    prediction: Optional[str] = None
    extracted_answer: Any = None
    is_correct: Optional[bool] = None
    confidence: Optional[float] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExampleResult":
        """Create from dict."""
        return cls(
            example_id=data["example_id"],
            question=data["question"],
            ground_truth=data["ground_truth"],
            prediction=data.get("prediction"),
            extracted_answer=data.get("extracted_answer"),
            is_correct=data.get("is_correct"),
            confidence=data.get("confidence"),
            raw_response=data.get("raw_response"),
            error=data.get("error"),
        )


class Checkpoint:
    """
    Manages experiment checkpoints for resumption.
    
    Saves results incrementally to a JSONL file.
    On resume, loads completed examples and skips them.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        enabled: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.enabled = enabled
        self._completed: dict[str, ExampleResult] = {}
        self._file_handle = None

    def load(self) -> dict[str, ExampleResult]:
        """
        Load existing checkpoint if available.
        
        Returns:
            Dict mapping example_id to ExampleResult for completed examples.
        """
        if not self.enabled:
            return {}
        
        if not self.checkpoint_path.exists():
            return {}
        
        self._completed = {}
        with open(self.checkpoint_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    result = ExampleResult.from_dict(data)
                    self._completed[result.example_id] = result
                except (json.JSONDecodeError, KeyError) as e:
                    # Skip malformed lines but continue
                    continue
        
        return self._completed

    def is_completed(self, example_id: str) -> bool:
        """Check if an example has been completed."""
        return example_id in self._completed

    def get_result(self, example_id: str) -> Optional[ExampleResult]:
        """Get cached result for an example."""
        return self._completed.get(example_id)

    def save_result(self, result: ExampleResult) -> None:
        """
        Append a result to the checkpoint file.
        
        Writes immediately for durability.
        """
        if not self.enabled:
            return
        
        self._completed[result.example_id] = result
        
        # Append to file
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    def get_completed_results(self) -> list[ExampleResult]:
        """Get all completed results."""
        return list(self._completed.values())

    def get_completed_ids(self) -> set[str]:
        """Get IDs of all completed examples."""
        return set(self._completed.keys())

    def num_completed(self) -> int:
        """Number of completed examples."""
        return len(self._completed)

    def clear(self) -> None:
        """Clear checkpoint (for fresh start)."""
        self._completed = {}
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

    def iter_results(self) -> Iterator[ExampleResult]:
        """Iterate over completed results."""
        return iter(self._completed.values())

    def __len__(self) -> int:
        return len(self._completed)

    def __contains__(self, example_id: str) -> bool:
        return example_id in self._completed

