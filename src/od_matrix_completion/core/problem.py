from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from.dto import Dimensions
from .base_algo import BaseOptimizer, OptimizationResult


def _safe_log_ratio(D: np.ndarray, D_hat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Безопасно вычислить log(D / D_hat) покомпонентно (для градиента KL)."""

    Dc = np.maximum(D, eps)
    Hc = np.maximum(D_hat, eps)
    return np.log(Dc / Hc)





class Problem:
    """Задача восстановления OD-матрицы (линейная модель f ≈ A @ d).

    Что хранит и делает:
      - Данные: A, наблюдаемые счётчики f_obs, веса датчиков sensor_weights,
        априорная матрица D_hat, маргиналии L (строки) и W (столбцы), γ.
      - Проверяет согласованность размерностей.
      - Помогает с векторизацией d = vec(D) и обратно.
      - Считает целевую и градиент для линейного случая.
      - Вызывает переданный алгоритм-оптимизатор.
    """

    def __init__(
        self,
        *,
        A: Optional[NDArray[np.float64]] = None,
        f_obs: Optional[np.ndarray] = None,
        sensor_weights: Optional[NDArray[np.float64]] = None,
        D_hat: Optional[NDArray[np.float64]] = None,
        L: Optional[NDArray[np.float64]] = None,
        W: Optional[NDArray[np.float64]] = None,
        gamma: float = 0.0,
    ) -> None:
        self.A = None if A is None else np.asarray(A, dtype=float)
        self.f_obs = None if f_obs is None else np.asarray(f_obs, dtype=float).reshape(-1)
        self.sensor_weights = None if sensor_weights is None else np.asarray(sensor_weights, dtype=float).reshape(-1)
        self.D_hat = None if D_hat is None else np.asarray(D_hat, dtype=float)
        self.L = None if L is None else np.asarray(L, dtype=float).reshape(-1)
        self.W = None if W is None else np.asarray(W, dtype=float).reshape(-1)
        self.gamma = float(gamma)

        self._algo: Optional[BaseOptimizer] = None

    def _infer_dimensions(self) -> Dimensions:
        if self.A is None:
            raise ValueError("A (routing matrix) must be provided to infer dimensions")

        m, n = self.A.shape
        n_zones = None
        # Prefer explicit hints from D_hat/L/W if available
        if self.D_hat is not None:
            if self.D_hat.ndim != 2 or self.D_hat.shape[0] != self.D_hat.shape[1]:
                raise ValueError("D_hat must be a square 2D array")
            n_zones = self.D_hat.shape[0]
            if n != n_zones * n_zones:
                raise ValueError(
                    f"A has {n} columns, but D_hat implies {n_zones*n_zones} OD pairs"
                )
        elif self.L is not None and self.W is not None:
            if self.L.size != self.W.size:
                raise ValueError("L and W must have the same length (n_zones)")
            n_zones = int(self.L.size)
            if n != n_zones * n_zones:
                raise ValueError(
                    f"A has {n} columns, but marginals imply {n_zones*n_zones} OD pairs"
                )
        else:
            root = int(round(np.sqrt(n)))
            if root * root != n:
                raise ValueError(
                    "Cannot infer n_zones: A's column count is not a perfect square, "
                    "and neither D_hat nor (L, W) were provided"
                )
            n_zones = root

        return Dimensions(n_edges=m, n_zones=n_zones, n_od_pairs=n)

    def validate(self) -> Dimensions:
        dims = self._infer_dimensions()

        if self.f_obs is not None and self.f_obs.size != dims.n_edges:
            raise ValueError(
                f"f_obs must have length {dims.n_edges}, got {self.f_obs.size}"
            )
        if self.sensor_weights is not None and self.sensor_weights.size != dims.n_edges:
            raise ValueError(
                f"sensor_weights must have length {dims.n_edges}, got {self.sensor_weights.size}"
            )
        if self.L is not None and self.L.size != dims.n_zones:
            raise ValueError(
                f"L must have length {dims.n_zones}, got {self.L.size}"
            )
        if self.W is not None and self.W.size != dims.n_zones:
            raise ValueError(
                f"W must have length {dims.n_zones}, got {self.W.size}"
            )
        if self.D_hat is not None and self.D_hat.shape != (dims.n_zones, dims.n_zones):
            raise ValueError(
                f"D_hat must have shape {(dims.n_zones, dims.n_zones)}, got {self.D_hat.shape}"
            )

        return dims

    def vec(self, D: np.ndarray) -> np.ndarray:
        D = np.asarray(D, dtype=float)
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError("D must be a square 2D array")
        return D.reshape(-1)

    def unvec(self, d: np.ndarray) -> np.ndarray:
        dims = self._infer_dimensions()
        d = np.asarray(d, dtype=float).reshape(-1)
        if d.size != dims.n_od_pairs:
            raise ValueError(
                f"d must have length {dims.n_od_pairs}, got {d.size}"
            )
        return d.reshape(dims.n_zones, dims.n_zones)

    # --------------------- Линейная модель -------------------------------
    def linear_prediction(self, d: np.ndarray) -> np.ndarray:
        if self.A is None:
            raise ValueError("A must be set for linear prediction")
        return self.A @ np.asarray(d, dtype=float).reshape(-1)

    def weighted_residual(self, d: np.ndarray) -> np.ndarray:
        if self.f_obs is None:
            raise ValueError("f_obs must be set to compute residuals")
        resid = self.linear_prediction(d) - self.f_obs
        if self.sensor_weights is None:
            return resid
        return self.sensor_weights * resid

    def objective(self, x: np.ndarray | np.matrix | None) -> float:
        """Целевая функция (линейный случай).

        0.5 * || W^{1/2} (A d - f_obs) ||^2 + gamma * KL(D || D_hat)
        Если часть данных отсутствует (например, f_obs или D_hat), соответствующий
        слагаемый пропускается.
        """

        dims = self.validate()
        val = 0.0

        if x is None:
            raise ValueError("x must be provided as d or D")

        if x.ndim == 1:
            d = np.asarray(x, dtype=float)
            D = d.reshape(dims.n_zones, dims.n_zones)
        elif x.ndim == 2:
            D = np.asarray(x, dtype=float)
            d = D.reshape(-1)
        else:
            raise ValueError("x must be 1D (d) or 2D square (D)")

        # Невязка по счетчикам
        if self.f_obs is not None:
            resid = self.linear_prediction(d) - self.f_obs
            if self.sensor_weights is not None:
                val += 0.5 * float(np.dot(self.sensor_weights * resid, resid))
            else:
                val += 0.5 * float(resid @ resid)

        # Регуляризация по KL
        if self.gamma > 0.0 and self.D_hat is not None:
            # KL(D||D_hat) = sum d_ij * log(d_ij / dh_ij) - d_ij + dh_ij
            eps = 1e-12
            Dc = np.maximum(D, eps)
            Hc = np.maximum(self.D_hat, eps)
            kl = float(np.sum(Dc * np.log(Dc / Hc) - Dc + Hc))
            val += self.gamma * kl

        return val

    def gradient_linear(self, d: np.ndarray) -> np.ndarray:
        """Градиент линейной цели по d = vec(D).

        grad = A^T W (A d - f_obs) + gamma * vec(log(D / D_hat))
        Второе слагаемое добавляется, только если задана D_hat и gamma > 0.
        """

        dims = self.validate()
        d = np.asarray(d, dtype=float).reshape(-1)

        # Первое слагаемое: A^T W (A d - f)
        resid = self.linear_prediction(d) - (self.f_obs if self.f_obs is not None else 0.0)
        if self.sensor_weights is not None:
            grad = self.A.T @ (self.sensor_weights * resid)
        else:
            grad = self.A.T @ resid

        # Добавка от KL, если включена
        if self.gamma > 0.0 and self.D_hat is not None:
            D = d.reshape(dims.n_zones, dims.n_zones)
            grad += self.gamma * _safe_log_ratio(D, self.D_hat).reshape(-1)

        return grad

    # ------------------------ Инициализация D ----------------------------
    def initial_guess(self) -> np.ndarray:
        """
        Начальное приблиэение для матрицы корреспонденций

        Returns:
            np.ndarray: _description_
        """

        dims = self._infer_dimensions()

        if self.D_hat is not None:
            return np.maximum(self.D_hat, 0.0).astype(float, copy=True)

        if self.L is not None and self.W is not None:
            seed = np.ones((dims.n_zones, dims.n_zones), dtype=float)
            return self._ipf_to_marginals(seed, self.L, self.W)

        return np.ones((dims.n_zones, dims.n_zones), dtype=float)

    @staticmethod
    def _ipf_to_marginals(
        D0: np.ndarray,
        L: np.ndarray,
        W: np.ndarray,
        *,
        max_iters: int = 1000,
        tol: float = 1e-10,
        eps: float = 1e-15,
    ) -> np.ndarray:
        """IPF-проекция на заданные маргиналии (строки/столбцы).

        Мультипликативно масштабирует строки/столбцы:
        D_{i,:} *= L_i / sum_j D_{i,j}, D_{:,j} *= W_j / sum_i D_{i,j}.
        Возвращает новую матрицу.
        """

        D = np.maximum(np.asarray(D0, dtype=float), 0.0)
        L = np.asarray(L, dtype=float).reshape(-1)
        W = np.asarray(W, dtype=float).reshape(-1)

        if D.shape != (L.size, W.size):
            raise ValueError("D0 shape must match (len(L), len(W))")

        for _ in range(max_iters):
            # Масштабирование строк
            row_sums = D.sum(axis=1)
            row_scale = L / np.maximum(row_sums, eps)
            D *= row_scale[:, None]

            # Масштабирование столбцов
            col_sums = D.sum(axis=0)
            col_scale = W / np.maximum(col_sums, eps)
            D *= col_scale[None, :]

            # Критерий остановки по маргиналиям
            if (
                np.allclose(D.sum(axis=1), L, rtol=0.0, atol=tol)
                and np.allclose(D.sum(axis=0), W, rtol=0.0, atol=tol)
            ):
                break

        return D

    # ------------------------------ Запуск --------------------------------
    def solve(
        self,
        *,
        algorithm: Optional[BaseOptimizer] = None,
        x0: Optional[np.ndarray] = None,
        callback: Optional[callable] = None,
    ) -> OptimizationResult:
        """Запускает переданный оптимизатор на этой задаче.

        Если ``algorithm`` не задан, используется тот, что был установлен
        заранее через ``set_algorithm(optimizer)``.
        """

        # Ранняя проверка размерностей
        dims = self.validate()
        _ = dims  # unused variable, validation is the main point

        algo = algorithm or self._algo
        if algo is None:
            raise RuntimeError("Не задан оптимизатор. Передайте его в solve(...) или вызовите set_algorithm(...)")
  

        return algo.fit(self, x0=x0, callback=callback)

    def set_algorithm(self, algo: BaseOptimizer) -> None:
        """Установить оптимизатор (экземпляр класса-наследника BaseOptimizer)."""

        if not isinstance(algo, BaseOptimizer):
            raise TypeError("Ожидается экземпляр класса-наследника BaseOptimizer")
        self._algo = algo
