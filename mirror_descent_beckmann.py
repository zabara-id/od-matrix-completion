from __future__ import annotations

from typing import Callable, Dict, Literal, Optional, Tuple

import numpy as np

from optimize_results_dto import MirrorDescentResult
from src.od_matrix_completion.core.models.manyalli_written_beckmann import (
    BRP,
    CSRGraph,
    fw_beckmann,
    fw_beckmann_flow,
)
from src.od_matrix_completion.core.models.soft_beckmann import fw_beckmann_soft

MirrorMode = Literal["hard", "soft_grad", "auto_soft"]


def project_to_marginals_masked(
    D: np.ndarray,
    L: np.ndarray,
    W: np.ndarray,
    allowed: np.ndarray,
    n_iters: int = 50,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    IPF/Sinkhorn-подобная проекция на заданные маргиналии (L, W) с сохранением структурных нулей.
    """
    D_proj = np.maximum(D, 0.0) * allowed

    for _ in range(n_iters):
        row_sum = D_proj.sum(axis=1, keepdims=True)
        row_scale = np.divide(L[:, None], row_sum, out=np.ones_like(row_sum), where=row_sum > eps)
        D_proj *= row_scale
        D_proj *= allowed

        col_sum = D_proj.sum(axis=0, keepdims=True)
        col_scale = np.divide(W[None, :], col_sum, out=np.ones_like(col_sum), where=col_sum > eps)
        D_proj *= col_scale
        D_proj *= allowed

    return D_proj


def kl_value_and_grad(
    D: np.ndarray,
    D_prior: np.ndarray,
    allowed: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[float, np.ndarray]:
    """
    Обобщённая KL дивергенция и её градиент по D с маской структурных нулей.
    """
    Dp = np.maximum(D, eps)
    Pp = np.maximum(D_prior, eps)
    log_ratio = np.log(Dp) - np.log(Pp)
    kl_mat = Dp * log_ratio - Dp + Pp
    kl_val = float(np.sum(allowed * kl_mat))
    kl_grad = allowed * log_ratio
    np.fill_diagonal(kl_grad, 0.0)
    return kl_val, kl_grad


def _apply_mask(vec: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return vec
    return vec * mask


def mirror_descent_beckmann(
    csr: CSRGraph,
    edge_cost: BRP,
    f_hat: np.ndarray,
    *,
    D_reference: np.ndarray,
    mode: MirrorMode = "hard",
    mask: Optional[np.ndarray] = None,
    reg_lambda: float = 0.0,
    D_prior: Optional[np.ndarray] = None,
    D_init: Optional[np.ndarray] = None,
    n_iters: int = 30,
    fw_hard_kwargs: Optional[Dict] = None,
    fw_soft_kwargs: Optional[Dict] = None,
    step0: float = 1e-2,
    ls_beta: float = 0.5,
    ls_min: float = 1e-12,
    ls_max_trials: int = 50,
    improve_eps: float = 1e-12,
    kl_eps: float = 1e-12,
    stall_iters: int = 3,
    stall_tol: float = 1e-4,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    switch_callback: Optional[Callable[[int], None]] = None,
) -> MirrorDescentResult:
    """
    Зеркальный спуск по OD-матрице с KL-регуляризацией.

    Режимы:
      - "hard": потоки и градиент через обычный Бекманн;
      - "soft_grad": потоки считаются по обычному Бекманну, градиент берётся из soft Beckmann;
      - "auto_soft": начинаем как "hard", переключаем градиент на soft если objective стагнирует `stall_iters` шагов.
    """
    if mode not in {"hard", "soft_grad", "auto_soft"}:
        raise ValueError(f"Unknown mode={mode}")

    D_reference = np.asarray(D_reference, dtype=np.float64)
    n = D_reference.shape[0]
    allowed = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(allowed, 0.0)

    L_ref = D_reference.sum(axis=1)
    W_ref = D_reference.sum(axis=0)
    ref_l1 = float(np.sum(np.abs(D_reference * allowed)))

    if fw_hard_kwargs is None:
        fw_hard_kwargs = {"max_iter": 30, "rgap_target": 1e-3, "verbose": False, "use_numba": True}
    if fw_soft_kwargs is None:
        fw_soft_kwargs = {
            "max_iter": 30,
            "theta": 10.0,
            "delta_rel": 0.02,
            "delta_abs": 1e-3,
            "verbose": False,
            "use_numba": True,
        }

    f_hat = np.asarray(f_hat, dtype=np.float64)
    if f_hat.shape[0] != csr.m:
        raise ValueError(f"f_hat must have shape ({csr.m},), got {f_hat.shape}")
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64)
        if mask.shape != f_hat.shape:
            raise ValueError(f"mask shape {mask.shape} must match flow shape {f_hat.shape}")

    if D_prior is None:
        rng = np.random.default_rng(0)
        noise = 0.005 * rng.standard_normal(D_reference.shape)
        D_prior = D_reference * (1.0 + noise)
        D_prior = np.maximum(D_prior, kl_eps)
        np.fill_diagonal(D_prior, 0.0)
        D_prior = project_to_marginals_masked(D_prior, L_ref, W_ref, allowed, n_iters=30)
    else:
        D_prior = np.asarray(D_prior, dtype=np.float64)
    if D_prior.shape != (n, n):
        raise ValueError(f"D_prior must have shape {(n, n)}, got {D_prior.shape}")
    D_prior = np.maximum(D_prior, 0.0) * allowed

    if reg_lambda > 0.0 and np.any(D_prior <= 0.0):
        # для KL избегаем нулей в prior
        D_prior = np.maximum(D_prior, kl_eps)
        D_prior *= allowed

    if D_init is None:
        D_est = np.outer(L_ref, W_ref) / max(W_ref.sum(), 1e-12)
        D_est = project_to_marginals_masked(D_est, L_ref, W_ref, allowed, n_iters=80)
    else:
        D_est = np.asarray(D_init, dtype=np.float64)
        if D_est.shape != (n, n):
            raise ValueError(f"D_init must have shape {(n, n)}, got {D_est.shape}")
        D_est = np.maximum(D_est, 0.0) * allowed
        D_est = project_to_marginals_masked(D_est, L_ref, W_ref, allowed, n_iters=50)

    def rel_l1(D: np.ndarray) -> float:
        diff = float(np.sum(np.abs((D - D_reference) * allowed)))
        return diff / max(ref_l1, 1e-12)

    def objective_value(D: np.ndarray) -> Tuple[float, float, np.ndarray]:
        flow_val = fw_beckmann_flow(csr, edge_cost, D, **fw_hard_kwargs)
        residual = _apply_mask(flow_val - f_hat, mask)
        data = 0.5 * float(np.dot(residual, residual))
        kl_val, _ = kl_value_and_grad(D, D_prior, allowed, eps=kl_eps)
        return data + reg_lambda * kl_val, kl_val, flow_val

    def evaluate_with_gradient(D: np.ndarray, use_soft_grad: bool):
        kl_val, kl_grad = kl_value_and_grad(D, D_prior, allowed, eps=kl_eps)

        if use_soft_grad:
            flow_hard = fw_beckmann_flow(csr, edge_cost, D, **fw_hard_kwargs)
            residual = _apply_mask(flow_hard - f_hat, mask)
            data = 0.5 * float(np.dot(residual, residual))
            _, JT = fw_beckmann_soft(csr, edge_cost, D, **fw_soft_kwargs)
            grad_data = JT(residual)
            grad = grad_data + reg_lambda * kl_grad
            grad_source = "soft"
        else:
            flow_hard, jac = fw_beckmann(csr, edge_cost, D, **fw_hard_kwargs)
            residual = _apply_mask(flow_hard - f_hat, mask)
            data = 0.5 * float(np.dot(residual, residual))
            grad_data = (jac.T @ residual).reshape(D.shape)
            grad = grad_data + reg_lambda * kl_grad
            grad_source = "hard"

        obj = data + reg_lambda * kl_val
        return obj, grad, flow_hard, kl_val, grad_source

    # параметры зеркального шага
    eps_floor = 1e-12
    exp_clip = 50.0
    projection_iters = 30

    use_soft_grad = mode == "soft_grad"
    switch_iter: int | None = None

    # стартовая оценка
    obj, grad, flow_current, kl_val, grad_source = evaluate_with_gradient(D_est, use_soft_grad)

    objective_history = [obj]
    kl_history = [kl_val]
    rel_l1_history = [rel_l1(D_est)]
    gradient_sources = [grad_source]

    for it in range(n_iters):
        if progress_callback is not None:
            progress_callback(it, n_iters, mode)
        step = step0
        accepted = False
        best_D = D_est
        best_obj = obj
        best_kl = kl_val
        best_flow = flow_current
        trials_used = 0

        for t in range(ls_max_trials):
            trials_used = t + 1
            base = np.maximum(D_est, eps_floor) * allowed
            expo = np.clip(-step * grad, -exp_clip, exp_clip)
            candidate = base * np.exp(expo)
            candidate = project_to_marginals_masked(candidate, L_ref, W_ref, allowed, n_iters=projection_iters)

            cand_obj, cand_kl, cand_flow = objective_value(candidate)
            if cand_obj < obj - improve_eps:
                best_obj = cand_obj
                best_D = candidate
                best_kl = cand_kl
                best_flow = cand_flow
                accepted = True
                break

            step *= ls_beta
            if step < ls_min:
                break

        D_est = best_D
        obj = best_obj
        flow_current = best_flow
        kl_val = best_kl

        if accepted and trials_used <= 2:
            step0 = min(step0 * 1.3, 1e-1)
        elif not accepted:
            step0 *= 0.5

        # пересчитываем градиент для следующего шага (вдруг сменился режим)
        obj_eval, grad, flow_current, kl_val_eval, grad_source = evaluate_with_gradient(D_est, use_soft_grad)
        # обновляем запись, чтобы истории шли от одной и той же оценки
        obj = obj_eval
        kl_val = kl_val_eval

        objective_history.append(obj)
        kl_history.append(kl_val)
        rel_l1_history.append(rel_l1(D_est))
        gradient_sources.append(grad_source)

        # переключение режима после стагнации на окне из stall_iters итераций
        if mode == "auto_soft" and not use_soft_grad:
            if len(objective_history) >= stall_iters + 1:
                recent = objective_history[-(stall_iters + 1):]
                span = max(recent) - min(recent)
                if span <= stall_tol * max(1.0, abs(recent[-1])):
                    use_soft_grad = True
                    switch_iter = it + 1  # включили soft grad после этой итерации
                    if switch_callback is not None:
                        switch_callback(switch_iter)
                    else:
                        print(f"[auto_soft] переключение на soft-град на итерации {switch_iter}")

    times_final = edge_cost(flow_current)

    return MirrorDescentResult(
        mode=mode,
        D_final=D_est,
        flow_final=flow_current,
        times_final=times_final,
        objective_history=objective_history,
        rel_l1_history=rel_l1_history,
        kl_history=kl_history,
        gradient_sources=gradient_sources,
        switch_iter=switch_iter,
    )
