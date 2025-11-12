from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
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


class BaseOptimizer(ABC):
    """Базовый интерфейс для алгоритмов оптимизации.
    """

    def __init__(
        self,
        *,
        max_iters: int = 1000,
        tol: float = 1e-6,
        verbose: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        """Конструктор класса.

        Args:
            max_iters (int, optional): Максимальное число итераций. По умолчанию 1000.
            tol (float, optional): Точность. По умолчанию 1e-6.
            verbose (bool, optional): Отладка (колбэк). По умолчанию False.
            random_state (Optional[int], optional): _description_. По умолчанию None.
        """
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.verbose = bool(verbose)
        self.random_state = random_state

    @abstractmethod
    def fit(
        self,
        problem: "Problem",
        x0: Optional[NDArray[np.float64]] = None,
        callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> OptimizationResult:
        """Абсирактный метод для запуска алгоритма оптимизации.    

        Args:
            problem (Problem): экземпляр задачи Problem;
            x0 (Optional[NDArray[np.float64]], optional): начальное приближение (d или D). По умолчанию None.
            callback (Optional[Callable[[dict[str, Any]], None]], optional): колбэк для отладки. По умолчанию None.

        Raises:
            NotImplementedError: не реализованный алгоритм.

        Returns:
            OptimizationResult: DTO с результатами оптимизации.
        """
        
        raise NotImplementedError
