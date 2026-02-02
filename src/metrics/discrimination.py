"""Discrimination metrics: AUROC, AUPRC, selective prediction."""

from typing import Sequence

import numpy as np


def auroc(
    confidences: Sequence[float],
    correctness: Sequence[bool],
) -> float:
    """
    Compute Area Under ROC Curve.
    
    Measures the ability of confidence to distinguish correct from
    incorrect predictions. Higher confidence should correlate with
    correct predictions.
    
    Args:
        confidences: Model confidence scores.
        correctness: Binary labels (True = correct).
    
    Returns:
        AUROC in [0, 1]. 0.5 = random, 1.0 = perfect discrimination.
    """
    if len(confidences) == 0:
        return 0.5
    
    confidences = np.array(confidences)
    correctness = np.array(correctness, dtype=bool)
    
    n_correct = correctness.sum()
    n_incorrect = len(correctness) - n_correct
    
    if n_correct == 0 or n_incorrect == 0:
        return 0.5
    
    # Sort by confidence (descending)
    sorted_idx = np.argsort(-confidences)
    correctness_sorted = correctness[sorted_idx]
    
    # Compute AUC via Wilcoxon-Mann-Whitney statistic
    # AUC = P(confidence(correct) > confidence(incorrect))
    correct_ranks = np.where(correctness_sorted)[0]
    
    # Sum of ranks of correct predictions
    rank_sum = (len(correctness) - correct_ranks).sum()
    
    # Normalize
    auc = (rank_sum - n_correct * (n_correct + 1) / 2) / (n_correct * n_incorrect)
    
    return float(auc)


def auprc(
    confidences: Sequence[float],
    correctness: Sequence[bool],
) -> float:
    """
    Compute Area Under Precision-Recall Curve.
    
    More appropriate than AUROC when classes are imbalanced.
    
    Args:
        confidences: Model confidence scores.
        correctness: Binary labels.
    
    Returns:
        AUPRC in [0, 1].
    """
    if len(confidences) == 0:
        return 0.0
    
    confidences = np.array(confidences)
    correctness = np.array(correctness, dtype=bool)
    
    # Sort by confidence (descending)
    sorted_idx = np.argsort(-confidences)
    correctness_sorted = correctness[sorted_idx]
    
    # Compute precision and recall at each threshold
    tp_cumsum = np.cumsum(correctness_sorted)
    n_predictions = np.arange(1, len(correctness) + 1)
    
    precision = tp_cumsum / n_predictions
    recall = tp_cumsum / correctness.sum() if correctness.sum() > 0 else np.zeros_like(tp_cumsum)
    
    # Compute AUC using trapezoidal rule
    # Prepend (0, 1) point for proper area calculation
    recall = np.concatenate([[0], recall])
    precision = np.concatenate([[1], precision])
    
    # AUC
    auprc_value = np.trapz(precision, recall)
    
    return float(abs(auprc_value))


def accuracy_at_coverage(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    coverage: float = 0.8,
) -> tuple[float, float]:
    """
    Compute accuracy when keeping only top-confidence predictions.
    
    Useful for selective prediction: abstain on low-confidence examples.
    
    Args:
        confidences: Model confidence scores.
        correctness: Binary labels.
        coverage: Fraction of examples to keep (0, 1].
    
    Returns:
        Tuple of (accuracy_on_kept, threshold).
        - accuracy_on_kept: Accuracy on the kept examples.
        - threshold: Confidence threshold used.
    """
    if len(confidences) == 0:
        return 0.0, 0.0
    
    confidences = np.array(confidences)
    correctness = np.array(correctness, dtype=bool)
    
    # Determine threshold for desired coverage
    n_keep = int(len(confidences) * coverage)
    n_keep = max(1, n_keep)
    
    sorted_conf = np.sort(confidences)[::-1]
    threshold = sorted_conf[n_keep - 1]
    
    # Keep examples with confidence >= threshold
    mask = confidences >= threshold
    
    # Handle ties by keeping exactly n_keep
    if mask.sum() > n_keep:
        indices = np.argsort(-confidences)[:n_keep]
        mask = np.zeros_like(mask)
        mask[indices] = True
    
    if mask.sum() == 0:
        return 0.0, threshold
    
    accuracy = correctness[mask].mean()
    
    return float(accuracy), float(threshold)


def selective_risk_curve(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute selective risk curve: risk vs coverage.
    
    Risk = 1 - accuracy = error rate.
    
    Args:
        confidences: Model confidence scores.
        correctness: Binary labels.
        n_points: Number of points on the curve.
    
    Returns:
        Tuple of (coverages, risks) arrays.
    """
    if len(confidences) == 0:
        return np.array([]), np.array([])
    
    coverages = np.linspace(0.01, 1.0, n_points)
    risks = []
    
    for cov in coverages:
        acc, _ = accuracy_at_coverage(confidences, correctness, cov)
        risks.append(1.0 - acc)
    
    return coverages, np.array(risks)


def area_under_selective_risk(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    n_points: int = 100,
) -> float:
    """
    Compute Area Under Selective Risk Curve (AUSRC).
    
    Lower is better - indicates better selective prediction.
    """
    coverages, risks = selective_risk_curve(confidences, correctness, n_points)
    
    if len(coverages) == 0:
        return 1.0
    
    return float(np.trapz(risks, coverages))

