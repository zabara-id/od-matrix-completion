

def safe_log_ratio(D: np.ndarray, D_hat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Безопасно вычислить log(D / D_hat) покомпонентно (для градиента KL)."""

    Dc = np.maximum(D, eps)
    Hc = np.maximum(D_hat, eps)
    return np.log(Dc / Hc)