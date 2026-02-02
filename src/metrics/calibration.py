"""Calibration metrics: ECE, Brier score, reliability diagrams."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE = sum_{b} (n_b / N) * |acc_b - conf_b|
    
    where b indexes bins, n_b is bin size, acc_b is accuracy in bin,
    and conf_b is mean confidence in bin.
    
    Args:
        confidences: Model confidence scores in [0, 1].
        correctness: Binary labels (True = correct).
        n_bins: Number of bins for calibration.
        strategy: "uniform" (equal width) or "quantile" (equal size).
    
    Returns:
        ECE value in [0, 1]. Lower is better.
    """
    if len(confidences) == 0:
        return 0.0
    
    confidences = np.array(confidences)
    correctness = np.array(correctness, dtype=float)
    n = len(confidences)
    
    if strategy == "uniform":
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        # Equal-sized bins based on confidence quantiles
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_boundaries = np.percentile(confidences, quantiles)
        bin_boundaries[0] = 0.0
        bin_boundaries[-1] = 1.0
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    ece = 0.0
    
    for i in range(n_bins):
        # Find samples in this bin
        if i == n_bins - 1:
            # Last bin includes upper boundary
            mask = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        else:
            mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        
        bin_size = mask.sum()
        
        if bin_size > 0:
            bin_confidence = confidences[mask].mean()
            bin_accuracy = correctness[mask].mean()
            ece += (bin_size / n) * abs(bin_accuracy - bin_confidence)
    
    return float(ece)


def brier_score(
    confidences: Sequence[float],
    correctness: Sequence[bool],
) -> float:
    """
    Compute Brier score.
    
    Brier = (1/N) * sum((confidence - correctness)^2)
    
    Args:
        confidences: Model confidence scores in [0, 1].
        correctness: Binary labels (True = correct).
    
    Returns:
        Brier score in [0, 1]. Lower is better.
    """
    if len(confidences) == 0:
        return 0.0
    
    confidences = np.array(confidences)
    correctness = np.array(correctness, dtype=float)
    
    return float(np.mean((confidences - correctness) ** 2))


@dataclass
class ReliabilityBin:
    """Data for a single bin in a reliability diagram."""
    bin_lower: float
    bin_upper: float
    mean_confidence: float
    accuracy: float
    count: int
    gap: float  # accuracy - mean_confidence


def reliability_diagram_data(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """
    Compute data for a reliability diagram.
    
    Args:
        confidences: Model confidence scores.
        correctness: Binary labels.
        n_bins: Number of bins.
    
    Returns:
        List of ReliabilityBin objects, one per bin.
    """
    if len(confidences) == 0:
        return []
    
    confidences = np.array(confidences)
    correctness = np.array(correctness, dtype=float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bins = []
    
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        
        count = int(mask.sum())
        
        if count > 0:
            mean_conf = float(confidences[mask].mean())
            acc = float(correctness[mask].mean())
        else:
            mean_conf = (lower + upper) / 2
            acc = 0.0
        
        bins.append(ReliabilityBin(
            bin_lower=lower,
            bin_upper=upper,
            mean_confidence=mean_conf,
            accuracy=acc,
            count=count,
            gap=acc - mean_conf,
        ))
    
    return bins


def maximum_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    
    MCE = max_b |acc_b - conf_b|
    
    Args:
        confidences: Model confidence scores.
        correctness: Binary labels.
        n_bins: Number of bins.
    
    Returns:
        MCE value in [0, 1].
    """
    bins = reliability_diagram_data(confidences, correctness, n_bins)
    
    if not bins:
        return 0.0
    
    return max(abs(b.gap) for b in bins if b.count > 0)


def adaptive_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """
    Compute Adaptive Calibration Error (ACE).
    
    Like ECE but uses quantile-based (equal-sized) bins.
    """
    return expected_calibration_error(
        confidences,
        correctness,
        n_bins=n_bins,
        strategy="quantile",
    )

