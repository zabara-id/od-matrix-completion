from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class OptimizationResult:
    """Результат работы оптимизатора.

    Параметры:
      - D: восстановленная OD-матрица (n_zones x n_zones)
      - d: векторизация D (строчная, длина n_zones * n_zones)
      - objective: значение целевой функции в конце (если есть)
      - n_iters: число итераций
      - converged: сошёлся ли метод
      - history: список метрик по итерациям (произвольно)
    """

    D: np.ndarray
    d: np.ndarray
    objective: Optional[float] = None
    n_iters: int = 0
    converged: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)


class BaseOptimizer(ABC):
    """Базовый интерфейс для алгоритмов оптимизации OD-матрицы.

    Ориентирован на линейный случай: f ≈ A @ d, где d = vec(D).
    Конкретные алгоритмы должны реализовать метод ``fit``.
    """

    def __init__(
        self,
        *,
        max_iters: int = 1000,
        tol: float = 1e-6,
        verbose: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.verbose = bool(verbose)
        self.random_state = random_state

    def supports(self, problem: "Problem") -> bool:  # noqa: F821 (forward ref)
        """Можно ли решать данную задачу этим оптимизатором.

        По умолчанию — да. Переопределяйте при необходимости.
        """

        return True

    @abstractmethod
    def fit(
        self,
        problem: "Problem",  # noqa: F821 (forward ref)
        x0: Optional[np.ndarray] = None,
        callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> OptimizationResult:
        """Запуск оптимизации и возврат результата.

        Аргументы:
          - problem: экземпляр задачи Problem
          - x0: начальное приближение (d или D). Если None — алгоритм сам выберет.
          - callback: опциональный колбэк с метриками по итерациям
        """

        raise NotImplementedError
