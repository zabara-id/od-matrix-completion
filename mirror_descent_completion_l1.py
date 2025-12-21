from __future__ import annotations

from typing import Callable, Dict, Literal, Optional, Tuple

import numpy as np

from mirror_descent_completion_kl import project_to_marginals_masked
from optimize_results_l1_dto import MirrorDescentL1Result
from src.od_matrix_completion.core.models.manyalli_written_beckmann import (
    BRP,
    CSRGraph,
    fw_beckmann,
    fw_beckmann_flow,
)
from src.od_matrix_completion.core.models.soft_beckmann import fw_beckmann_soft


MirrorMode = Literal["hard", "soft_grad", "auto_soft"]


def _apply_mask(vec: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return vec
    return vec * mask


def l1_to_true_value_and_subgrad(
    D: np.ndarray,
    D_true: np.ndarray,
    allowed: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    ||D - D_true||_1 и субградиент по D (с маской структурных нулей).
    """
    diff = (D - D_true) * allowed
    val = float(np.sum(np.abs(diff)))
    subgrad = allowed * np.sign(D - D_true)
    np.fill_diagonal(subgrad, 0.0)
    return val, subgrad


def mirror_descent_completion_l1(
    csr: CSRGraph,
    edge_cost: BRP,
    f_hat: np.ndarray,
    *,
    D_reference: np.ndarray,
    D_true: np.ndarray,
    mode: MirrorMode = "hard",
    mask: Optional[np.ndarray] = None,
    reg_lambda: float = 0.0,
    D_init: Optional[np.ndarray] = None,
    n_iters: int = 30,
    fw_hard_kwargs: Optional[Dict] = None,
    fw_soft_kwargs: Optional[Dict] = None,
    step0: float = 1e-2,
    ls_beta: float = 0.5,
    ls_min: float = 1e-12,
    ls_max_trials: int = 50,
    improve_eps: float = 1e-12,
    stall_iters: int = 3,
    stall_tol: float = 1e-4,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    switch_callback: Optional[Callable[[int], None]] = None,
) -> MirrorDescentL1Result:
    """
    Зеркальный спуск для восстановления OD-матрицы по наблюдаемым потокам в модели Бекмана
    с L1-регуляризацией к истинной матрице:

        objective(D) = data(D) + λ · ||D - D_true||_1,

    где data(D) = 0.5 * ||mask ⊙ (f(D) - f_hat)||_2^2.

    В качестве зеркального шага используется энтропийная геометрия (multiplicative update),
    затем проекция на фиксированные маргиналии (из D_reference) с сохранением нулевой диагонали.
    """
    np.random.seed(42)
    if mode not in {"hard", "soft_grad", "auto_soft"}:
        raise ValueError(f"Unknown mode={mode}")

    D_reference = np.asarray(D_reference, dtype=np.float64)
    D_true = np.asarray(D_true, dtype=np.float64)
    n = int(D_reference.shape[0])
    if D_reference.shape != (n, n):
        raise ValueError(f"D_reference must have shape {(n, n)}, got {D_reference.shape}")
    if D_true.shape != (n, n):
        raise ValueError(f"D_true must have shape {(n, n)}, got {D_true.shape}")

    allowed = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(allowed, 0.0)

    L_ref = D_true.sum(axis=1)
    W_ref = D_true.sum(axis=0)
    ref_l1 = float(np.sum(np.abs(D_reference * allowed)))
    true_l1 = float(np.sum(np.abs(D_true * allowed)))

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
    if f_hat.shape != (csr.m,):
        raise ValueError(f"f_hat must have shape ({csr.m},), got {f_hat.shape}")
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64)
        if mask.shape != f_hat.shape:
            raise ValueError(f"mask shape {mask.shape} must match flow shape {f_hat.shape}")

    # D_reference = project_to_marginals_masked(D_reference, L_ref, W_ref, allowed)

    if D_init is None:
        u, Sigma, v = np.linalg.svd(D_true)
        Sigma[0] *= 0.2
        D_est = project_to_marginals_masked(u @ np.diag(Sigma) @ v, L_ref, W_ref, allowed)

        # D_est = D_reference * (1 + 70*np.random.random(D_reference.shape))
        # D_est = project_to_marginals_masked(D_est, L_ref, W_ref, allowed)

        print("D_est - D_true = ", np.linalg.norm(D_est - D_true, ord=1) / np.linalg.norm(D_true, ord=1))

        print("D_reference - D_true = ", np.linalg.norm(D_reference - D_true, ord=1) / np.linalg.norm(D_reference, ord=1))
    else:
        D_est = np.asarray(D_init, dtype=np.float64)
        if D_est.shape != (n, n):
            raise ValueError(f"D_init must have shape {(n, n)}, got {D_est.shape}")
        D_est = np.maximum(D_est, 0.0) * allowed
        D_est = project_to_marginals_masked(D_est, L_ref, W_ref, allowed)

    def rel_l1_ref(D: np.ndarray) -> float:
        diff = float(np.sum(np.abs((D - D_reference) * allowed)))
        return diff / max(ref_l1, 1e-12)

    def rel_l1_true(D: np.ndarray) -> float:
        diff = float(np.sum(np.abs((D - D_true) * allowed)))
        return diff / max(true_l1, 1e-12)

    def objective_value(D: np.ndarray) -> Tuple[float, float, np.ndarray]:
        flow_val = fw_beckmann_flow(csr, edge_cost, D, **fw_hard_kwargs)
        residual = _apply_mask(flow_val - f_hat, mask)
        data = 0.5 * float(np.dot(residual, residual))
        l1_val, _ = l1_to_true_value_and_subgrad(D, D_true, allowed)
        return data + reg_lambda * l1_val, l1_val, flow_val

    def evaluate_with_gradient(D: np.ndarray, use_soft_grad: bool):
        l1_val, l1_subgrad = l1_to_true_value_and_subgrad(D, D_true, allowed)

        if use_soft_grad:
            flow_hard = fw_beckmann_flow(csr, edge_cost, D, **fw_hard_kwargs)
            residual = _apply_mask(flow_hard - f_hat, mask)
            data = 0.5 * float(np.dot(residual, residual))
            _, JT = fw_beckmann_soft(csr, edge_cost, D, **fw_soft_kwargs)
            grad_data = JT(residual)
            grad = grad_data + reg_lambda * l1_subgrad
            grad_source = "soft"
        else:
            flow_hard, jac = fw_beckmann(csr, edge_cost, D, **fw_hard_kwargs)
            residual = _apply_mask(flow_hard - f_hat, mask)
            data = 0.5 * float(np.dot(residual, residual))
            grad_data = (jac.T @ residual).reshape(D.shape)
            grad = grad_data + reg_lambda * l1_subgrad
            grad_source = "hard"

        obj = data + reg_lambda * l1_val
        return obj, grad, flow_hard, l1_val, grad_source

    eps_floor = 1e-12
    exp_clip = 50.0
    projection_iters = 30

    use_soft_grad = mode == "soft_grad"
    switch_iter: int | None = None

    obj, grad, flow_current, l1_val, grad_source = evaluate_with_gradient(D_est, use_soft_grad)
    grad_norm = float(np.linalg.norm(grad))

    objective_history = [obj]
    l1_to_true_history = [l1_val]
    rel_l1_ref_history = [rel_l1_ref(D_est)]
    rel_l1_true_history = [rel_l1_true(D_est)]
    gradient_sources = [grad_source]
    data_history = [np.linalg.norm(flow_current - f_hat, ord=1) / max(np.linalg.norm(f_hat, ord=1), 1e-12)]
    residual_full0 = flow_current - f_hat
    data_full_history = [0.5 * float(np.dot(residual_full0, residual_full0))]
    step_history: list[float] = []
    ls_trials_history: list[int] = []
    accepted_history: list[bool] = []
    grad_norm_history: list[float] = [grad_norm]

    for it in range(int(n_iters)):
        if progress_callback is not None:
            progress_callback(it, n_iters, mode)

        step = float(step0)
        step_used = step
        accepted = False
        best_D = D_est
        best_obj = obj
        best_l1 = l1_val
        best_flow = flow_current
        trials_used = 0

        for t in range(int(ls_max_trials)):
            trials_used = t + 1
            base = np.maximum(D_est, eps_floor) * allowed
            expo = np.clip(-step * grad, -exp_clip, exp_clip)
            candidate = base * np.exp(expo)
            candidate = project_to_marginals_masked(candidate, L_ref, W_ref, allowed, n_iters=projection_iters)

            cand_obj, cand_l1, cand_flow = objective_value(candidate)
            if cand_obj < obj - improve_eps:
                best_obj = cand_obj
                best_D = candidate
                best_l1 = cand_l1
                best_flow = cand_flow
                accepted = True
                step_used = step
                break

            step *= float(ls_beta)
            step_used = step
            if step < float(ls_min):
                break

        D_est = best_D
        obj = best_obj
        flow_current = best_flow
        l1_val = best_l1

        if accepted and trials_used <= 2:
            step0 = min(float(step0) * 1.3, 1e-1)
        elif not accepted:
            step0 = float(step0) * 0.5

        obj_eval, grad, flow_current, l1_val_eval, grad_source = evaluate_with_gradient(D_est, use_soft_grad)
        grad_norm = float(np.linalg.norm(grad))
        obj = obj_eval
        l1_val = l1_val_eval

        objective_history.append(obj)
        l1_to_true_history.append(l1_val)
        rel_l1_ref_history.append(rel_l1_ref(D_est))
        rel_l1_true_history.append(rel_l1_true(D_est))
        gradient_sources.append(grad_source)
        data_history.append(np.linalg.norm(flow_current - f_hat, ord=1) / max(np.linalg.norm(f_hat, ord=1), 1e-12))
        residual_full = flow_current - f_hat
        data_full_history.append(0.5 * float(np.dot(residual_full, residual_full)))
        step_history.append(float(step_used))
        ls_trials_history.append(int(trials_used))
        accepted_history.append(bool(accepted))
        grad_norm_history.append(grad_norm)

        if mode == "auto_soft" and not use_soft_grad:
            if len(objective_history) >= int(stall_iters) + 1:
                recent = objective_history[-(int(stall_iters) + 1) :]
                span = max(recent) - min(recent)
                if span <= float(stall_tol) * max(1.0, abs(recent[-1])):
                    use_soft_grad = True
                    switch_iter = it + 1
                    if switch_callback is not None:
                        switch_callback(switch_iter)
                    else:
                        print(f"[auto_soft] переключение на soft-град на итерации {switch_iter}")

    times_final = edge_cost(flow_current)

    return MirrorDescentL1Result(
        mode=mode,
        D_final=D_est,
        flow_final=flow_current,
        times_final=times_final,
        objective_history=objective_history,
        rel_l1_ref_history=rel_l1_ref_history,
        l1_to_true_history=l1_to_true_history,
        rel_l1_true_history=rel_l1_true_history,
        gradient_sources=gradient_sources,
        data_history=data_history,
        data_full_history=data_full_history,
        step_history=step_history,
        ls_trials_history=ls_trials_history,
        accepted_history=accepted_history,
        grad_norm_history=grad_norm_history,
        switch_iter=switch_iter,
    )

