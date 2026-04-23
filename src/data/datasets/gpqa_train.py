"""GPQA (train): difference set gpqa_main \\ gpqa_diamond (N=250)."""

from typing import Optional

from datasets import load_dataset
from dotenv import load_dotenv

from src.data.base import DataExample
from src.data.datasets.gpqa import GPQADataset

# Load HF_TOKEN (and any other env vars) from .env if present. GPQA is a gated
# dataset, so authentication is required to download it.
load_dotenv()


class GPQATrainDataset(GPQADataset):
    """
    GPQA train split, defined as the difference set
    ``gpqa_main`` (N=448) minus ``gpqa_diamond`` (N=198), giving N=250.

    Uses the ``Record ID`` column from HuggingFace to identify overlapping
    instances between the two configs (gpqa_diamond is a subset of gpqa_main).
    All question formatting, answer extraction, and verification logic is
    inherited from ``GPQADataset``.
    """

    def __init__(
        self,
        split: str = "train",  # GPQA only has a train split on HF
        max_examples: Optional[int] = None,
        seed: Optional[int] = None,
        hf_path: str = "Idavidrein/gpqa",
        superset_name: str = "gpqa_main",
        exclude_name: str = "gpqa_diamond",
    ):
        super().__init__(
            split=split,
            max_examples=max_examples,
            seed=seed,
            hf_path=hf_path,
            hf_name=superset_name,
        )
        self.superset_name = superset_name
        self.exclude_name = exclude_name

    @property
    def name(self) -> str:
        return "gpqa-train"

    def _load_examples(self) -> list[DataExample]:
        """Load ``superset_name`` and drop rows whose Record ID appears in ``exclude_name``."""
        superset = load_dataset(self.hf_path, self.superset_name, split=self.split)
        exclude_set = load_dataset(self.hf_path, self.exclude_name, split=self.split)

        exclude_ids = {item["Record ID"] for item in exclude_set}

        # Keep ``i`` as the index in the superset so that the deterministic
        # choice-shuffling (seeded by ``self.seed + i``) is stable even when
        # the exclusion set changes.
        id_prefix = f"gpqa_{self.superset_name}_minus_{self.exclude_name}_{self.split}"

        examples: list[DataExample] = []
        for i, item in enumerate(superset):
            if item["Record ID"] in exclude_ids:
                continue
            examples.append(self._build_example(item, i, id_prefix=id_prefix))
        return examples
