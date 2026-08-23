"""Reusable isotonic calibration artifacts for confidence probes."""

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.metrics.custom import c_star_metrics


ISOTONIC_REGRESSION_FILENAME = "isotonic_regression.json"
ISOTONIC_REGRESSION_ARTIFACT_VERSION = 1


def _as_finite_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {vector.shape}.")
    if len(vector) == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def confidence_metrics(
    predictions: Sequence[float],
    targets: Sequence[float],
) -> dict[str, float]:
    """Compute the confidence metrics persisted with calibration artifacts."""
    metrics = c_star_metrics(predictions, targets)
    return {
        "mae": metrics.mae,
        "mse": metrics.mse,
        "rmse": metrics.rmse,
        "pearson_r": metrics.pearson_r,
        "spearman_r": metrics.spearman_r,
        "ece": metrics.ece,
    }


def apply_isotonic_regression(
    predictions: Sequence[float],
    artifact: dict[str, Any],
) -> np.ndarray:
    """Apply a serialized isotonic mapping with clipped out-of-bounds values."""
    values = _as_finite_vector(predictions, name="predictions")
    x_thresholds = _as_finite_vector(
        artifact.get("X_thresholds_", []),
        name="X_thresholds_",
    )
    y_thresholds = _as_finite_vector(
        artifact.get("y_thresholds_", []),
        name="y_thresholds_",
    )
    if x_thresholds.shape != y_thresholds.shape:
        raise ValueError(
            "Isotonic X_thresholds_ and y_thresholds_ must have matching shapes."
        )
    if np.any(np.diff(x_thresholds) < 0):
        raise ValueError("Isotonic X_thresholds_ must be sorted.")
    if np.any(np.diff(y_thresholds) < 0):
        raise ValueError("Isotonic y_thresholds_ must be non-decreasing.")

    calibrated = np.interp(
        values,
        x_thresholds,
        y_thresholds,
        left=y_thresholds[0],
        right=y_thresholds[-1],
    )
    y_min = float(artifact.get("y_min", 0.0))
    y_max = float(artifact.get("y_max", 1.0))
    return np.clip(calibrated, y_min, y_max)


def fit_isotonic_regression(
    predictions: Sequence[float],
    targets: Sequence[float],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit an isotonic mapping and return a JSON-serializable artifact."""
    raw_predictions = _as_finite_vector(predictions, name="predictions")
    target_values = _as_finite_vector(targets, name="targets")
    if raw_predictions.shape != target_values.shape:
        raise ValueError("Isotonic predictions and targets must have matching shapes.")
    if np.any((raw_predictions < 0.0) | (raw_predictions > 1.0)):
        raise ValueError("Isotonic predictions must be in [0, 1].")
    if np.any((target_values < 0.0) | (target_values > 1.0)):
        raise ValueError("Isotonic targets must be in [0, 1].")

    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(raw_predictions, target_values)

    artifact: dict[str, Any] = {
        "type": "isotonic_regression",
        "artifact_version": ISOTONIC_REGRESSION_ARTIFACT_VERSION,
        "out_of_bounds": "clip",
        "y_min": 0.0,
        "y_max": 1.0,
        "n_train_examples": int(len(raw_predictions)),
        "X_thresholds_": [float(value) for value in model.X_thresholds_],
        "y_thresholds_": [float(value) for value in model.y_thresholds_],
    }
    if metadata:
        artifact["metadata"] = metadata

    calibrated_predictions = apply_isotonic_regression(raw_predictions, artifact)
    artifact["training_metrics"] = {
        "raw": confidence_metrics(raw_predictions, target_values),
        "calibrated": confidence_metrics(calibrated_predictions, target_values),
    }
    return artifact


def save_isotonic_regression(
    artifact: dict[str, Any],
    checkpoint_dir: str | Path,
) -> Path:
    """Save an isotonic sidecar in a probe checkpoint directory."""
    output_path = Path(checkpoint_dir) / ISOTONIC_REGRESSION_FILENAME
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return output_path


def load_isotonic_regression(
    checkpoint_dir: str | Path,
) -> dict[str, Any] | None:
    """Load an isotonic sidecar when one exists."""
    artifact_path = Path(checkpoint_dir) / ISOTONIC_REGRESSION_FILENAME
    if not artifact_path.exists():
        return None
    artifact = json.loads(artifact_path.read_text())
    if artifact.get("type") != "isotonic_regression":
        raise ValueError(f"Unknown isotonic artifact type in {artifact_path}.")
    return artifact
