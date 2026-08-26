"""Versioned isotonic mappings from unrestricted scalar scores to probabilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.metrics.custom import c_star_metrics

from .artifacts import atomic_write_json


ISOTONIC_ARTIFACT_VERSION = 2
ISOTONIC_FILENAME = "isotonic_regression.json"


def _vector(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or len(result) == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return result


def metrics(predictions: Sequence[float], targets: Sequence[float]) -> dict[str, float]:
    value = c_star_metrics(predictions, targets)
    return {
        "mae": value.mae,
        "mse": value.mse,
        "rmse": value.rmse,
        "pearson_r": value.pearson_r,
        "spearman_r": value.spearman_r,
        "ece": value.ece,
    }


def fit_isotonic_scores(
    scores: Sequence[float],
    targets: Sequence[float],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score_values = _vector(scores, "scores")
    target_values = _vector(targets, "targets")
    if score_values.shape != target_values.shape:
        raise ValueError("scores and targets must have matching shapes")
    if np.any((target_values < 0.0) | (target_values > 1.0)):
        raise ValueError("targets must be in [0, 1]")
    model = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
        y_min=0.0,
        y_max=1.0,
    )
    calibrated = model.fit_transform(score_values, target_values)
    raw_probabilities = 1.0 / (1.0 + np.exp(-score_values))
    artifact: dict[str, Any] = {
        "type": "isotonic_regression",
        "artifact_version": ISOTONIC_ARTIFACT_VERSION,
        "input_type": "score",
        "increasing": True,
        "out_of_bounds": "clip",
        "y_min": 0.0,
        "y_max": 1.0,
        "n_validation_examples": int(len(scores)),
        "X_thresholds_": [float(value) for value in model.X_thresholds_],
        "y_thresholds_": [float(value) for value in model.y_thresholds_],
        "validation_metrics": {
            "raw": metrics(raw_probabilities, target_values),
            "calibrated": metrics(calibrated, target_values),
        },
    }
    if metadata:
        artifact["metadata"] = metadata
    return artifact


def apply_isotonic(values: Sequence[float], artifact: dict[str, Any]) -> np.ndarray:
    source = _vector(values, "values")
    version = int(artifact.get("artifact_version", 1))
    input_type = artifact.get("input_type", "probability" if version == 1 else "score")
    if version not in (1, 2):
        raise ValueError(f"Unsupported isotonic artifact version: {version}")
    if input_type not in ("score", "probability"):
        raise ValueError(f"Unknown isotonic input_type: {input_type}")
    if input_type == "probability" and np.any((source < 0.0) | (source > 1.0)):
        raise ValueError("Version-1 isotonic input probabilities must be in [0, 1]")
    x_values = _vector(artifact["X_thresholds_"], "X_thresholds_")
    y_values = _vector(artifact["y_thresholds_"], "y_thresholds_")
    if x_values.shape != y_values.shape:
        raise ValueError("Isotonic threshold arrays must have matching shapes")
    if np.any(np.diff(x_values) < 0.0) or np.any(np.diff(y_values) < 0.0):
        raise ValueError("Isotonic thresholds must be non-decreasing")
    calibrated = np.interp(
        source,
        x_values,
        y_values,
        left=y_values[0],
        right=y_values[-1],
    )
    return np.clip(
        calibrated,
        float(artifact.get("y_min", 0.0)),
        float(artifact.get("y_max", 1.0)),
    )


def save_isotonic(artifact: dict[str, Any], checkpoint_dir: str | Path) -> Path:
    path = Path(checkpoint_dir) / ISOTONIC_FILENAME
    atomic_write_json(path, artifact)
    return path


def load_isotonic(path_or_directory: str | Path) -> dict[str, Any]:
    path = Path(path_or_directory)
    if path.is_dir():
        path = path / ISOTONIC_FILENAME
    artifact = json.loads(path.read_text())
    if artifact.get("type") != "isotonic_regression":
        raise ValueError(f"Invalid isotonic artifact: {path}")
    return artifact

