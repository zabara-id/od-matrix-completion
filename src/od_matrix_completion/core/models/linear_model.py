from __future__ import annotations

import numpy as np


def _safe_log_ratio(D: np.ndarray, H: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    Dc = np.maximum(D, eps)
    Hc = np.maximum(H, eps)
    return np.log(Dc / Hc)


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
    gamma: float = 0.0,
    D_hat: np.ndarray | None = None,
    n_zones: int | None = None,
) -> np.ndarray:
    """Градиент линейной цели по d = vec(D).

    grad = A^T W (A d - f_obs) + gamma * vec(log(D / D_hat))
    Второе слагаемое добавляется, только если задана D_hat и gamma > 0.
    """

    A = np.asarray(A, dtype=float)
    d = np.asarray(d, dtype=float).reshape(-1)

    resid = linear_prediction(A, d) - (f_obs if f_obs is not None else 0.0)
    if sensor_weights is not None:
        grad = A.T @ (sensor_weights * resid)
    else:
        grad = A.T @ resid

    if gamma > 0.0 and D_hat is not None:
        if n_zones is None:
            n = A.shape[1]
            root = int(round(np.sqrt(n)))
            if root * root != n:
                raise ValueError("Cannot infer n_zones from A; please pass n_zones explicitly")
            n_zones = root
        D = d.reshape(n_zones, n_zones)
        grad = grad + float(gamma) * _safe_log_ratio(D, np.asarray(D_hat, dtype=float)).reshape(-1)

    return grad
