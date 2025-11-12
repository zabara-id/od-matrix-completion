from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np

from .base_algo import BaseOptimizer, OptimizationResult
from .problem import Problem


class KLProximalDescent(BaseOptimizer):
    """Проксимальный (зеркальный) спуск в KL-геометрии с мультипликативным апдейтом.

    Задача: min_D g(D) + gamma * KL(D || D_hat), где g(D) = 0.5||W^{1/2}(A vec(D) - f)||^2.

    Итерация (компонентно):
        log D_{k+1} = (gamma * log D_hat + (1/eta_k) * log D_k - grad_g(D_k)) / (gamma + 1/eta_k)
      или при gamma=0: D_{k+1} = D_k * exp(-eta_k * grad_g(D_k))  (exponentiated gradient)

    После апдейта можно проецировать на маргиналии (IPF), что согласуется с KL-геометрией.

    Параметры:
      - step_size: базовый шаг eta
      - schedule: "constant" | "sqrt" | "linear" (затухание шага)
      - project_marginals: включать IPF-проекцию (если заданы L и W)
      - ipf_iters: число итераций IPF за шаг
      - eps: нижняя грань для положительности (во избежание log(0))
    """

    def __init__(
        self,
        *,
        step_size: float = 1e-2,
        schedule: str = "constant",
        project_marginals: bool = True,
        ipf_iters: int = 2,
        eps: float = 1e-12,
        max_iters: int = 500,
        tol: float = 1e-6,
        verbose: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            max_iters=max_iters,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
        )
        self.step_size = float(step_size)
        self.schedule = str(schedule).lower()
        self.project_marginals = bool(project_marginals)
        self.ipf_iters = int(ipf_iters)
        self.eps = float(eps)

    def _eta(self, k: int) -> float:
        if self.schedule == "constant":
            return self.step_size
        if self.schedule == "sqrt":
            return self.step_size / math.sqrt(k + 1)
        if self.schedule == "linear":
            return self.step_size / (k + 1)
        return self.step_size

    def _grad_data(self, problem: Problem, d: np.ndarray) -> np.ndarray:
        # градиент только по g(D): A^T W (A d - f)
        resid = problem.linear_prediction(d)
        if problem.f_obs is not None:
            resid = resid - problem.f_obs
        if problem.sensor_weights is not None:
            return problem.A.T @ (problem.sensor_weights * resid)
        return problem.A.T @ resid

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

        # Гарантируем положительность
        D = np.maximum(D, self.eps)
        d = D.reshape(-1)

        obj = problem.objective(D)
        best_D = D.copy()
        best_d = d.copy()
        best_obj = obj

        history: list[dict[str, Any]] = []
        converged = False

        H = None
        if problem.gamma > 0.0:
            if problem.D_hat is not None:
                H = np.maximum(problem.D_hat, self.eps)
            else:
                H = np.ones_like(D)

        for k in range(self.max_iters):
            eta = self._eta(k)
            d = D.reshape(-1)
            g = self._grad_data(problem, d).reshape(dims.n_zones, dims.n_zones)

            if problem.gamma > 0.0:
                denom = problem.gamma + 1.0 / max(eta, 1e-18)
                log_D = np.log(np.maximum(D, self.eps))
                log_H = np.log(H)
                # log D_{new} = (gamma*log H + (1/eta)*log D - g) / denom
                log_D_new = (problem.gamma * log_H + (1.0 / max(eta, 1e-18)) * log_D - g) / denom
                D_new = np.exp(log_D_new)
            else:
                # классический exponentiated gradient: D_new = D * exp(-eta * g)
                D_new = D * np.exp(-eta * g)

            # Проекция на маргиналии (KL-геометрия хорошо сочетается с IPF)
            if self.project_marginals and problem.L is not None and problem.W is not None:
                D_new = problem._ipf_to_marginals(np.maximum(D_new, self.eps), problem.L, problem.W, max_iters=self.ipf_iters)
            else:
                D_new = np.maximum(D_new, self.eps)

            d_new = D_new.reshape(-1)
            obj_new = problem.objective(D_new)

            step_norm = float(np.linalg.norm(d_new - d))
            rel_step = step_norm / max(1.0, float(np.linalg.norm(d)))

            info = {
                "iter": k + 1,
                "obj": obj_new,
                "step_norm": step_norm,
                "rel_step": rel_step,
                "eta": eta,
            }
            history.append(info)
            if callback is not None:
                try:
                    callback(info)
                except Exception:
                    pass

            if obj_new <= best_obj:
                best_obj = obj_new
                best_D = D_new.copy()
                best_d = d_new.copy()

            if self.verbose and (k == 0 or (k + 1) % 50 == 0):
                print(f"[ProxKL] it={k+1} obj={obj_new:.6e} step={rel_step:.3e} eta={eta:.3e}")

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

