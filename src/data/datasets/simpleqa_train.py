"""SimpleQA (train): difference set llamastack/simpleqa[train] \\ google/simpleqa-verified[eval]."""

import ast
from typing import Optional

from datasets import load_dataset
from dotenv import load_dotenv

from src.data.base import DataExample
from src.data.datasets.simpleqa_verified import SimpleQAVerifiedDataset

# Load HF_TOKEN etc. from .env if present (harmless if file is absent).
load_dotenv()


class SimpleQATrainDataset(SimpleQAVerifiedDataset):
    """
    SimpleQA train split, defined as the difference set:

        ``llamastack/simpleqa[train]`` (N=4326)
        minus ``google/simpleqa-verified[eval]`` (N=1000)
        = N=3326 train instances.

    The exclusion is performed on the integer ``original_index`` column of
    ``google/simpleqa-verified``, which points back to the row index of the
    ``llamastack/simpleqa`` train split. Note: ~56 rows in the verified set
    have revised ``problem`` text that no longer exactly matches the
    corresponding ``input_query`` in ``llamastack/simpleqa`` (manual question
    refinement during verification); they are still treated as the same row
    and excluded from train.

    All answer extraction / verification (string_match or llm_judge) logic is
    inherited from :class:`SimpleQAVerifiedDataset`.
    """

    def __init__(
        self,
        split: str = "train",  # llamastack/simpleqa only has a train split
        max_examples: Optional[int] = None,
        seed: Optional[int] = None,
        hf_path: str = "llamastack/simpleqa",
        exclude_hf_path: str = "google/simpleqa-verified",
        exclude_split: str = "eval",
        verification_mode: str = "string_match",  # or "llm_judge"
        llm_judge_model: Optional[str] = "openai/gpt-oss-20b",
        llm_judge_base_url: Optional[str] = None,
        llm_judge_max_concurrent: int = 1000,
    ):
        super().__init__(
            split=split,
            max_examples=max_examples,
            seed=seed,
            hf_path=hf_path,
            verification_mode=verification_mode,
            llm_judge_model=llm_judge_model,
            llm_judge_base_url=llm_judge_base_url,
            llm_judge_max_concurrent=llm_judge_max_concurrent,
        )
        self.exclude_hf_path = exclude_hf_path
        self.exclude_split = exclude_split

    @property
    def name(self) -> str:
        return "simpleqa-train"

    def _load_examples(self) -> list[DataExample]:
        """Load ``llamastack/simpleqa`` and drop rows whose index appears in the verified set's ``original_index``."""
        superset = load_dataset(self.hf_path, split=self.split)
        exclude_set = load_dataset(self.exclude_hf_path, split=self.exclude_split)

        exclude_indices: set[int] = {int(row["original_index"]) for row in exclude_set}

        examples: list[DataExample] = []
        for i, item in enumerate(superset):
            if i in exclude_indices:
                continue

            # ``metadata`` in llamastack/simpleqa is a stringified dict.
            raw_metadata = item.get("metadata", "")
            parsed_metadata: dict = {}
            if isinstance(raw_metadata, dict):
                parsed_metadata = raw_metadata
            elif isinstance(raw_metadata, str) and raw_metadata:
                try:
                    parsed_metadata = ast.literal_eval(raw_metadata)
                    if not isinstance(parsed_metadata, dict):
                        parsed_metadata = {}
                except (ValueError, SyntaxError):
                    parsed_metadata = {}

            # Keep ``i`` (the row's index in llamastack/simpleqa) as the id
            # suffix so ids are stable even if the exclusion set changes.
            examples.append(DataExample(
                id=f"simpleqa_train_{i}",
                question=item["input_query"],
                answer=item["expected_answer"],
                metadata={
                    "original_index": i,
                    "topic": parsed_metadata.get("topic", ""),
                    "answer_type": parsed_metadata.get("answer_type", ""),
                    "urls": parsed_metadata.get("urls", []),
                },
            ))

        return examples
