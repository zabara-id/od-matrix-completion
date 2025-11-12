from __future__ import annotations

import numpy as np


def linear_prediction(A: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Линейное предсказание f ≈ A @ d."""
    if A is None:
        raise ValueError("A must be set for linear prediction")
    return np.asarray(A, dtype=float) @ np.asarray(d, dtype=float).reshape(-1)


def gradient_linear(
    d: np.ndarray,
    *,
    A: np.ndarray,
    f_obs: np.ndarray | None = None,
    sensor_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Градиент только по датчику: A^T W (A d - f_obs). Без регуляризации."""

    A = np.asarray(A, dtype=float)
    d = np.asarray(d, dtype=float).reshape(-1)

    resid = linear_prediction(A, d) - (f_obs if f_obs is not None else 0.0)
    if sensor_weights is not None:
        return A.T @ (sensor_weights * resid)
    return A.T @ resid
