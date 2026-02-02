"""Custom metrics for c(x) calibration evaluation."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class CStarMetrics:
    """Metrics for evaluating c(x) against c*(x)."""
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    pearson_r: float  # Pearson correlation
    spearman_r: float  # Spearman rank correlation
    ece: float  # ECE treating c*(x) as ground truth
    num_examples: int


def c_star_metrics(
    c_estimates: Sequence[float],
    c_star: Sequence[float],
) -> CStarMetrics:
    """
    Compute metrics comparing c(x) estimates to c*(x) ground truth.
    
    This is specific to the "pre-generation confidence" setting where:
    - c(x) is the model's confidence estimate for question x
    - c*(x) is the ground truth accuracy from sampling k times
    
    Args:
        c_estimates: Model's c(x) confidence estimates.
        c_star: Ground truth c*(x) from sampling.
    
    Returns:
        CStarMetrics with various comparison metrics.
    """
    if len(c_estimates) == 0:
        return CStarMetrics(
            mae=0.0,
            mse=0.0,
            rmse=0.0,
            pearson_r=0.0,
            spearman_r=0.0,
            ece=0.0,
            num_examples=0,
        )
    
    c_est = np.array(c_estimates)
    c_gt = np.array(c_star)
    
    # Basic error metrics
    errors = c_est - c_gt
    mae = float(np.abs(errors).mean())
    mse = float((errors ** 2).mean())
    rmse = float(np.sqrt(mse))
    
    # Correlation metrics
    pearson_r = _pearson_correlation(c_est, c_gt)
    spearman_r = _spearman_correlation(c_est, c_gt)
    
    # ECE-style metric: does c(x) align with c*(x)?
    # Bin by c(x), compute mean c*(x) in each bin
    ece = _c_star_ece(c_est, c_gt)
    
    return CStarMetrics(
        mae=mae,
        mse=mse,
        rmse=rmse,
        pearson_r=pearson_r,
        spearman_r=spearman_r,
        ece=ece,
        num_examples=len(c_estimates),
    )


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    if len(x) < 2:
        return 0.0
    
    x_mean = x.mean()
    y_mean = y.mean()
    
    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = np.sqrt(((x - x_mean) ** 2).sum() * ((y - y_mean) ** 2).sum())
    
    if denominator == 0:
        return 0.0
    
    return float(numerator / denominator)


def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation coefficient."""
    if len(x) < 2:
        return 0.0
    
    # Convert to ranks
    x_ranks = _rank(x)
    y_ranks = _rank(y)
    
    return _pearson_correlation(x_ranks, y_ranks)


def _rank(x: np.ndarray) -> np.ndarray:
    """Compute ranks of values (1-based, average ties)."""
    sorted_idx = np.argsort(x)
    ranks = np.zeros_like(x, dtype=float)
    
    n = len(x)
    i = 0
    while i < n:
        j = i
        # Find all tied values
        while j < n - 1 and x[sorted_idx[j]] == x[sorted_idx[j + 1]]:
            j += 1
        # Assign average rank to all tied values
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    
    return ranks


def _c_star_ece(c_estimates: np.ndarray, c_star: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE-style metric for c(x) vs c*(x).
    
    Bins by c(x) and checks if mean c*(x) in each bin matches.
    """
    if len(c_estimates) == 0:
        return 0.0
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    n = len(c_estimates)
    ece = 0.0
    
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (c_estimates >= bin_boundaries[i]) & (c_estimates <= bin_boundaries[i + 1])
        else:
            mask = (c_estimates >= bin_boundaries[i]) & (c_estimates < bin_boundaries[i + 1])
        
        bin_size = mask.sum()
        
        if bin_size > 0:
            mean_c_est = c_estimates[mask].mean()
            mean_c_star = c_star[mask].mean()
            ece += (bin_size / n) * abs(mean_c_star - mean_c_est)
    
    return float(ece)


def compute_all_metrics(
    c_estimates: Sequence[float],
    c_star: Sequence[float],
    correctness: Sequence[bool] = None,
) -> dict:
    """
    Compute all relevant metrics for c(x) evaluation.
    
    Args:
        c_estimates: Model's c(x) estimates.
        c_star: Ground truth c*(x) from sampling.
        correctness: Optional binary correctness for single-shot evaluation.
    
    Returns:
        Dict of all metrics.
    """
    from .calibration import expected_calibration_error, brier_score
    from .discrimination import auroc
    
    metrics = {}
    
    # c*(x) comparison metrics
    c_metrics = c_star_metrics(c_estimates, c_star)
    metrics["c_star_mae"] = c_metrics.mae
    metrics["c_star_mse"] = c_metrics.mse
    metrics["c_star_rmse"] = c_metrics.rmse
    metrics["c_star_pearson"] = c_metrics.pearson_r
    metrics["c_star_spearman"] = c_metrics.spearman_r
    metrics["c_star_ece"] = c_metrics.ece
    
    # If binary correctness provided, compute standard calibration metrics
    if correctness is not None:
        metrics["ece"] = expected_calibration_error(c_estimates, correctness)
        metrics["brier"] = brier_score(c_estimates, correctness)
        metrics["auroc"] = auroc(c_estimates, correctness)
    
    return metrics

