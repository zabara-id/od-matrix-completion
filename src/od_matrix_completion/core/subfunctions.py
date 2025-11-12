import numpy as np
from numpy.typing import NDArray


def safe_log_ratio(
        D: NDArray[np.float64],
        D_hat: NDArray[np.float64],
        eps: float = 1e-12
) -> NDArray[np.float64]:
    
    Dc = np.maximum(D, eps)
    Hc = np.maximum(D_hat, eps)
    
    return np.log(Dc / Hc)