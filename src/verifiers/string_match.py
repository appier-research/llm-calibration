"""String matching verifier for exact/normalized answer comparison."""

import re
from typing import Any

from src.data.base import DataExample
from .base import BaseVerifier


class StringMatchVerifier(BaseVerifier):
    """
    Verifies answers by string matching.
    
    Supports:
    - Exact matching
    - Whitespace normalization
    - Case-insensitive matching
    - Numeric comparison with tolerance
    """

    def __init__(
        self,
        normalize_whitespace: bool = True,
        lowercase: bool = True,
        strip_punctuation: bool = False,
        numeric_tolerance: float = 1e-6,
        extract_numbers: bool = True,
    ):
        self.normalize_whitespace = normalize_whitespace
        self.lowercase = lowercase
        self.strip_punctuation = strip_punctuation
        self.numeric_tolerance = numeric_tolerance
        self.extract_numbers = extract_numbers

    @property
    def name(self) -> str:
        return "string_match"

    def verify(
        self,
        example: DataExample,
        response: Any,
    ) -> bool:
        """
        Check if response matches the ground truth answer.
        
        Args:
            example: Dataset example with ground truth.
            response: Extracted answer from model output.
        
        Returns:
            True if response matches ground truth.
        """
        ground_truth = example.answer
        
        # Handle None responses
        if response is None:
            return False
        
        # Try numeric comparison first if enabled
        if self.extract_numbers:
            gt_num = self._try_parse_number(ground_truth)
            resp_num = self._try_parse_number(response)
            
            if gt_num is not None and resp_num is not None:
                return abs(gt_num - resp_num) <= self.numeric_tolerance
        
        # Fall back to string comparison
        gt_normalized = self._normalize(str(ground_truth))
        resp_normalized = self._normalize(str(response))
        
        return gt_normalized == resp_normalized

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if self.normalize_whitespace:
            text = " ".join(text.split())
        
        if self.lowercase:
            text = text.lower()
        
        if self.strip_punctuation:
            text = re.sub(r"[^\w\s]", "", text)
        
        return text.strip()

    def _try_parse_number(self, value: Any) -> float | None:
        """Try to parse a value as a number."""
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove commas and whitespace
            cleaned = value.replace(",", "").strip()
            try:
                return float(cleaned)
            except ValueError:
                pass
        
        return None

