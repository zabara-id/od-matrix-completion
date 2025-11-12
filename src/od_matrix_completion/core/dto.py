from typing import Optional, Any
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class Dimensions:
    n_edges: int
    n_zones: int
    n_od_pairs: int


@dataclass(slots=True)
class OptimizationResult:
    """Результат работы алгоритма оптимизации.

    Args:
        D: восстановленная OD-матрица (n_zones x n_zones);
        d: векторизация D (строчная, длина n_zones * n_zones);
        objective: значение целевой функции в конце (если есть);
        n_iters: число итераций;
        converged: сошёлся ли метод;
        history: список метрик по итерациям (произвольно).
    """

    D: NDArray[np.float64]
    d: NDArray[np.float64]
    objective: Optional[float] = None
    n_iters: int = 0
    converged: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)