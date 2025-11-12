from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np

from .base_algo import BaseOptimizer, OptimizationResult
from .problem import Problem


class SubgradientDescent(BaseOptimizer):
    """Простой (суб)градиентный спуск для линейной постановки.

    Поддерживает:
      - постоянный или убывающий шаг (1/sqrt(k) или 1/k),
      - проекцию на неотрицательность,
      - IPF-проекцию на маргиналии (если заданы L и W).

    Параметры:
      - step_size: базовый шаг (eta)
      - schedule: "constant" | "sqrt" | "linear" (1/sqrt(k) или 1/k)
      - project_nonneg: проекция на D>=0 (обрезка снизу)
      - nonneg_floor: нижняя граница (обычно 0.0)
      - project_marginals: применять IPF-проекцию, если заданы L и W
      - ipf_iters: число итераций IPF за шаг
    """

    def __init__(
        self,
        *,
        step_size: float = 1e-2,
        schedule: str = "constant",
        project_nonneg: bool = True,
        nonneg_floor: float = 0.0,
        project_marginals: bool = True,
        ipf_iters: int = 2,
        max_iters: int = 500,
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            max_iters=max_iters,
            tol=tol,
            verbose=verbose,
        )
        self.step_size = float(step_size)
        self.schedule = str(schedule).lower()
        self.project_nonneg = bool(project_nonneg)
        self.nonneg_floor = float(nonneg_floor)
        self.project_marginals = bool(project_marginals)
        self.ipf_iters = int(ipf_iters)

    def _eta(self, k: int) -> float:
        if self.schedule == "constant":
            return self.step_size
        if self.schedule == "sqrt":
            return self.step_size / math.sqrt(k + 1)
        if self.schedule == "linear":
            return self.step_size / (k + 1)
        # по умолчанию — постоянный шаг
        return self.step_size

    def _project(self, problem: Problem, D: np.ndarray) -> np.ndarray:
        X = np.asarray(D, dtype=float)
        if self.project_nonneg:
            X = np.maximum(X, self.nonneg_floor)
        if self.project_marginals and problem.L is not None and problem.W is not None:
            X = problem._ipf_to_marginals(X, problem.L, problem.W, max_iters=self.ipf_iters)
        return X

    def fit(
        self,
        problem: Problem,
        x0: Optional[np.ndarray] = None,
        callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> OptimizationResult:
        dims = problem.validate()

        # Инициализация
        if x0 is None:
            D = problem.initial_guess()
        else:
            x0 = np.asarray(x0, dtype=float)
            if x0.ndim == 1:
                D = x0.reshape(dims.n_zones, dims.n_zones)
            elif x0.ndim == 2:
                if x0.shape != (dims.n_zones, dims.n_zones):
                    raise ValueError("x0 имеет некорректную форму (ожидается квадрат n_zones x n_zones)")
                D = x0
            else:
                raise ValueError("x0 должен быть вектором (d) или матрицей (D)")

        D = self._project(problem, D)
        d = D.reshape(-1)

        obj = problem.objective(D)
        best_D = D.copy()
        best_d = d.copy()
        best_obj = obj

        history: list[dict[str, Any]] = []
        converged = False

        for k in range(self.max_iters):
            # Градиент (линейный случай)
            g = problem.gradient_linear(d)
            grad_norm = float(np.linalg.norm(g))

            eta = self._eta(k)
            d_new = d - eta * g
            D_new = d_new.reshape(dims.n_zones, dims.n_zones)

            # Проекции
            D_new = self._project(problem, D_new)
            d_new = D_new.reshape(-1)

            # Оценка
            obj_new = problem.objective(D_new)

            step_norm = float(np.linalg.norm(d_new - d))
            rel_step = step_norm / max(1.0, float(np.linalg.norm(d)))

            # История/колбэк
            info = {
                "iter": k + 1,
                "obj": obj_new,
                "grad_norm": grad_norm,
                "step_norm": step_norm,
                "rel_step": rel_step,
                "eta": eta,
            }
            history.append(info)
            if callback is not None:
                try:
                    callback(info)
                except Exception:
                    # не прерываем оптимизацию из-за ошибок в колбэке
                    pass

            # Улучшаем лучшее
            if obj_new <= best_obj:
                best_obj = obj_new
                best_D = D_new.copy()
                best_d = d_new.copy()

            if self.verbose and (k == 0 or (k + 1) % 50 == 0):
                print(f"[SGD] it={k+1} obj={obj_new:.6e} |g|={grad_norm:.3e} step={rel_step:.3e} eta={eta:.3e}")

            # Критерии остановки: малый шаг или малое изменение 
            if rel_step <= self.tol or abs(obj_new - obj) <= self.tol * max(1.0, abs(obj)):
                converged = True
                D = D_new
                d = d_new
                obj = obj_new
                break

            D = D_new
            d = d_new
            obj = obj_new

        return OptimizationResult(
            D=best_D,
            d=best_d,
            objective=best_obj,
            n_iters=len(history),
            converged=converged,
            history=history,
        )
