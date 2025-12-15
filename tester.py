import csv
from pathlib import Path
from typing import Optional, Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np

from src.od_matrix_completion.core.models.manyalli_written_beckmann import (
    BRP,
    CSRGraph,
    fw_beckmann,
)

# Путь к матрице корреспонденций
OD_PATH = Path("data/processed/Mat_Car_ev.csv")


def load_od_matrix(path: Path = OD_PATH) -> np.ndarray:
    """Читает матрицу корреспонденций из TSV и возвращает np.ndarray с нулевой диагональю."""
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Файл {path} пустой")

    header = rows[0]
    data_rows = rows[1:]
    expected_width = len(header)

    matrix_rows = []
    for idx, row in enumerate(data_rows, start=1):
        if len(row) != expected_width:
            raise ValueError(
                f"Строка {idx} имеет {len(row)} столбцов вместо ожидаемых {expected_width}"
            )
        try:
            # Первый элемент — идентификатор зоны, пропускаем его
            matrix_rows.append([float(cell) for cell in row[1:]])
        except ValueError as exc:
            raise ValueError(f"Не удалось распарсить числа в строке {idx}") from exc

    mat = np.asarray(matrix_rows, dtype=np.float64)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"OD-матрица должна быть квадратной, получено {mat.shape}")

    # Корреспонденция из зоны в себя не используется
    np.fill_diagonal(mat, 0.0)
    return mat


def build_dense_graph(n_nodes: int) -> Tuple[CSRGraph, BRP]:
    """
    Строит связный ориентированный граф без петель с 4·n рёбрами.
    Каркас — кольцо (двунаправленное), сверху добавляются дальние дуги, чтобы усилить связанность.
    """
    if n_nodes < 3:
        raise ValueError("Для построения графа нужно минимум три вершины")

    target_edges = 4 * n_nodes
    max_edges = n_nodes * (n_nodes - 1)
    if target_edges > max_edges:
        raise ValueError(f"Нельзя построить граф без петель с {target_edges} рёбрами при n={n_nodes}")

    edges = []
    edge_set = set()

    def add_edge(u: int, v: int):
        if u == v:
            return
        if (u, v) in edge_set:
            return
        edge_set.add((u, v))
        edges.append((u, v))

    # Базовое кольцо (двустороннее) гарантирует сильную связность
    for i in range(n_nodes):
        j = (i + 1) % n_nodes
        add_edge(i, j)
        add_edge(j, i)

    # Добавляем дальние дуги, пока не наберём 4n уникальных рёбер
    step = 2
    while len(edges) < target_edges:
        for i in range(n_nodes):
            j = (i + step) % n_nodes
            add_edge(i, j)
            add_edge(j, i)
            if len(edges) >= target_edges:
                break
        step += 1
        if step > n_nodes + 2 and len(edges) < target_edges:
            break

    # Если всё ещё не хватает рёбер (например, при малых n), дозаполняем всеми оставшимися парами
    if len(edges) < target_edges:
        for u in range(n_nodes):
            for v in range(n_nodes):
                if len(edges) >= target_edges:
                    break
                add_edge(u, v)

    # Обрезаем лишние, если набрали чуть больше в последнем шаге
    edges = edges[:target_edges]
    tail = np.fromiter((u for u, _ in edges), dtype=np.int32, count=target_edges)
    head = np.fromiter((v for _, v in edges), dtype=np.int32, count=target_edges)
    if len(tail) != target_edges:
        raise RuntimeError("Не удалось построить граф с требуемым количеством рёбер")

    csr = CSRGraph.from_edges(n_nodes, tail, head)

    # Параметры рёбер: небольшая вариативность по емкости и свободному времени
    cap = 1800.0 + 200.0 * ((tail + head) % 4)
    t0 = 1.0 + 0.1 * ((np.abs(head - tail) % 5))
    alpha = np.full(csr.m, 0.15, dtype=np.float64)
    beta = np.full(csr.m, 4.0, dtype=np.float64)
    edge_cost = BRP(cap, t0, alpha, beta)

    return csr, edge_cost


def compute_flows_from_file(
    path: Path = OD_PATH, max_iter: int = 50, rgap_target: float = 1e-3
):
    """
    Читает OD-матрицу, строит связный граф и считает потоки на рёбрах по модели Бекмана.
    Возвращает CSR-граф, вектор потоков и времена проезда по рёбрам.
    """
    D = load_od_matrix(path)

    csr, edge_cost = build_dense_graph(D.shape[0])

    flow, _ = fw_beckmann(
        csr,
        edge_cost,
        D,
        max_iter=max_iter,
        rgap_target=rgap_target,
        verbose=False,
    )
    times = edge_cost(flow)
    return csr, flow, times


class LSProblem:
    """
    0.5 * || M (f(D) - f_hat) ||^2_2  +  reg_lambda * KL(D || D_prior)

    KL берём generalized KL (I-divergence) по i != j:
      KL(D||P) = sum_{i!=j} [ D_ij * log(D_ij / P_ij) - D_ij + P_ij ]
    Градиент по D: d/dD_ij = log(D_ij / P_ij).
    """

    def __init__(
        self,
        csr: CSRGraph,
        edge_cost: BRP,
        f_hat: np.ndarray,
        fw_kwargs: dict,
        mask: Optional[np.ndarray] = None,
        *,
        reg_kind: str = "none",          # "none" | "kl"
        reg_lambda: float = 0.0,
        D_prior: Optional[np.ndarray] = None,
        kl_eps: float = 1e-12,
        enforce_zero_diag_in_reg: bool = True,
    ):
        self.graph = csr
        self.edge_cost = edge_cost
        self.f_hat = np.asarray(f_hat, dtype=np.float64)
        self.fw_kwargs = fw_kwargs

        # mask (вес/маска по рёбрам)
        if mask is None:
            self.mask = None
        else:
            mask_arr = np.asarray(mask, dtype=np.float64)
            if mask_arr.shape != self.f_hat.shape:
                raise ValueError(
                    f"mask shape {mask_arr.shape} не совпадает с размером потоков {self.f_hat.shape}"
                )
            self.mask = mask_arr

        self.reg_kind = str(reg_kind).lower()
        self.reg_lambda = float(reg_lambda)
        if self.reg_lambda < 0.0:
            raise ValueError("reg_lambda must be >= 0")

        self.kl_eps = float(kl_eps)
        if self.kl_eps <= 0.0:
            raise ValueError("kl_eps must be > 0")

        self.enforce_zero_diag_in_reg = bool(enforce_zero_diag_in_reg)

        if self.reg_kind == "none" or self.reg_lambda == 0.0:
            self.D_prior = None
            self.kl_mask = None
        elif self.reg_kind == "kl":
            if D_prior is None:
                raise ValueError("D_prior must be provided when reg_kind='kl' and reg_lambda>0")
            self.D_prior = np.asarray(D_prior, dtype=np.float64)
            if self.D_prior.shape[0] != self.D_prior.shape[1]:
                raise ValueError(f"D_prior must be square, got {self.D_prior.shape}")

            # Маска для KL: исключаем диагональ, если она фиксируется в 0
            if self.enforce_zero_diag_in_reg:
                n = self.D_prior.shape[0]
                self.kl_mask = (np.ones((n, n), dtype=np.float64) - np.eye(n, dtype=np.float64))
            else:
                self.kl_mask = np.ones_like(self.D_prior, dtype=np.float64)
        else:
            raise ValueError(f"Unknown reg_kind={reg_kind}. Use 'none' or 'kl'.")

    def _apply_mask(self, residual: np.ndarray) -> np.ndarray:
        if self.mask is None:
            return residual
        return residual * self.mask

    def _kl_value_and_grad(self, D: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Возвращает (KL(D||P), grad_KL), где grad_KL = d/dD KL = log(D/P),
        всё с eps и маской по диагонали.
        """
        P = self.D_prior
        mask = self.kl_mask

        Dp = np.maximum(D, self.kl_eps)
        Pp = np.maximum(P, self.kl_eps)

        log_ratio = np.log(Dp) - np.log(Pp)  # log(D/P)

        # value: sum [ D log(D/P) - D + P ]
        kl_mat = Dp * log_ratio - Dp + Pp
        if mask is not None:
            kl_val = float(np.sum(mask * kl_mat))
            grad = mask * log_ratio
        else:
            kl_val = float(np.sum(kl_mat))
            grad = log_ratio

        return kl_val, grad

    def evaluate(self, D: np.ndarray) -> Tuple[float, np.ndarray]:
        D = np.asarray(D, dtype=np.float64)

        flow, grad = fw_beckmann(self.graph, self.edge_cost, D, **self.fw_kwargs)
        residual = self._apply_mask(flow - self.f_hat)

        value = 0.5 * float(np.dot(residual, residual))
        grad_D = (grad.T @ residual).reshape(D.shape)

        if self.reg_kind == "kl" and self.reg_lambda > 0.0:
            kl_val, kl_grad = self._kl_value_and_grad(D)
            value += self.reg_lambda * kl_val
            grad_D += self.reg_lambda * kl_grad

        return value, grad_D

    def residual_and_jacobian(self, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Возвращает остаток по потокам r = M (f(D) - f_hat) и якобиан dr/dD.
        """
        D = np.asarray(D, dtype=np.float64)
        flow, grad = fw_beckmann(self.graph, self.edge_cost, D, **self.fw_kwargs)
        residual = self._apply_mask(flow - self.f_hat)
        if self.mask is None:
            jac = grad
        else:
            jac = grad * self.mask[:, None]
        return residual, jac, flow

    def value_only(self, D: np.ndarray) -> float:
        D = np.asarray(D, dtype=np.float64)

        flow, _ = fw_beckmann(self.graph, self.edge_cost, D, **self.fw_kwargs)
        residual = self._apply_mask(flow - self.f_hat)
        value = 0.5 * float(np.dot(residual, residual))

        if self.reg_kind == "kl" and self.reg_lambda > 0.0:
            kl_val, _ = self._kl_value_and_grad(D)
            value += self.reg_lambda * kl_val

        return value


def save_objective_curve(obj_history, plot_path: Optional[Path], title: str) -> None:
    if plot_path is None:
        return
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.semilogy(obj_history, marker="o", ms=3)
    plt.xlabel("iteration")
    plt.ylabel("objective")
    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()



def project_to_marginals(D: np.ndarray, L: np.ndarray, W: np.ndarray, n_iters: int = 2) -> np.ndarray:
    """
    Итеративное пропорциональное взвешивание (IPF): проекция на заданные маргиналии L (строки) и W (столбцы).
    """
    D_proj = np.maximum(D, 0.0)
    for _ in range(n_iters):
        row_sum = D_proj.sum(axis=1, keepdims=True)
        row_scale = np.divide(L[:, None], row_sum, out=np.ones_like(row_sum), where=row_sum > 1e-12)
        D_proj *= row_scale

        col_sum = D_proj.sum(axis=0, keepdims=True)
        col_scale = np.divide(W[None, :], col_sum, out=np.ones_like(col_sum), where=col_sum > 1e-12)
        D_proj *= col_scale
    return D_proj


def project_to_marginals_masked(
    D: np.ndarray,
    L: np.ndarray,
    W: np.ndarray,
    allowed: np.ndarray,
    n_iters: int = 50,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    IPF/Sinkhorn-подобная проекция на заданные маргиналии (L, W)
    с сохранением структурных нулей: allowed[i,j]=0 => D[i,j] всегда 0.

    D, allowed: (n,n)
    L, W: (n,)
    """
    D_proj = np.maximum(D, 0.0) * allowed

    for _ in range(n_iters):
        # rows
        row_sum = D_proj.sum(axis=1, keepdims=True)
        row_scale = np.divide(L[:, None], row_sum, out=np.ones_like(row_sum), where=row_sum > eps)
        D_proj *= row_scale
        D_proj *= allowed

        # cols
        col_sum = D_proj.sum(axis=0, keepdims=True)
        col_scale = np.divide(W[None, :], col_sum, out=np.ones_like(col_sum), where=col_sum > eps)
        D_proj *= col_scale
        D_proj *= allowed

    return D_proj


def run_mirror_descent(
    csr,
    edge_cost,
    D_reference: np.ndarray,
    f_hat: np.ndarray,
    fw_eval_params: dict,
    *,
    mask: Optional[np.ndarray] = None,
    n_iters: int = 30,
    plot_path: Optional[Path] = None,
    # regularization (handled inside LSProblem)
    reg_kind: str = "none",          # "none" | "kl"
    reg_lambda: float = 0.0,
    D_prior: Optional[np.ndarray] = None,
) -> dict:
    """
    Mirror descent по OD-матрице с masked-IPF проекцией (структурный ноль на диагонали сохраняется).

    Требование: ваш LSProblem должен поддерживать (reg_kind, reg_lambda, D_prior) как в текущем коде.
    """
    D_reference = np.asarray(D_reference, dtype=np.float64)
    n = D_reference.shape[0]

    L_ref = D_reference.sum(axis=1)
    W_ref = D_reference.sum(axis=0)

    # allowed mask: запрещаем диагональ (OD ii = 0)
    allowed = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(allowed, 0.0)

    # prior по умолчанию для KL: IPF-матрица с теми же маргиналиями и структурным нулём на диагонали
    if reg_kind == "kl" and reg_lambda > 0.0 and D_prior is None:
        D_prior = np.outer(L_ref, W_ref) / max(W_ref.sum(), 1e-12)
        D_prior = project_to_marginals_masked(D_prior, L_ref, W_ref, allowed, n_iters=80)

    # ваш LSProblem (оставь как есть, только чтобы он принимал эти параметры)
    problem = LSProblem(
        csr,
        edge_cost,
        f_hat,
        fw_eval_params,
        mask=mask,
        reg_kind=reg_kind,
        reg_lambda=reg_lambda,
        D_prior=D_prior,
        kl_eps=1e-12,
        enforce_zero_diag_in_reg=True,
    )

    # Старт: IPF-аппроксимация с structural zeros
    D_est = np.outer(L_ref, W_ref) / max(W_ref.sum(), 1e-12)
    D_est = project_to_marginals_masked(D_est, L_ref, W_ref, allowed, n_iters=80)

    obj_history = []

    # Mirror Descent params
    step0 = 1e-2
    ls_beta = 0.5
    ls_min = 1e-12
    ls_max_trials = 50
    improve_eps = 1e-12

    eps_floor = 1e-12
    exp_clip = 50.0

    best_obj = float("inf")

    for k in range(n_iters):
        obj, subgrad_D = problem.evaluate(D_est)
        grad_norm = float(np.linalg.norm(subgrad_D))

        step = step0
        accepted = False
        best_iter_obj = obj
        best_iter_D = D_est
        trials_used = 0

        for t in range(ls_max_trials):
            trials_used = t + 1

            # mirror step
            base = np.maximum(D_est, eps_floor)
            expo = np.clip(-step * subgrad_D, -exp_clip, exp_clip)
            candidate = base * np.exp(expo)

            # enforce structural zeros via masked IPF
            candidate = project_to_marginals_masked(candidate, L_ref, W_ref, allowed, n_iters=30)

            obj_cand = problem.value_only(candidate)

            if obj_cand < obj - improve_eps:
                accepted = True
                best_iter_obj = obj_cand
                best_iter_D = candidate
                break

            step *= ls_beta
            if step < ls_min:
                break

        D_est = best_iter_D
        best_obj = min(best_obj, best_iter_obj)
        obj_history.append(best_iter_obj)

        if accepted and trials_used <= 2:
            step0 = min(step0 * 1.3, 1e-1)
        elif not accepted:
            step0 *= 0.5

        print(
            f"iter={k:03d} obj={best_iter_obj:.6e} best={best_obj:.6e} "
            f"grad_norm={grad_norm:.3e} step_used={step:.2e} accepted={accepted} trials={trials_used}"
        )

    flow_final, _ = fw_beckmann(csr, edge_cost, D_est, **fw_eval_params)
    times_final = edge_cost(flow_final)

    obj_ref = problem.value_only(D_reference)
    obj_est = problem.value_only(D_est)

    results = {
        "D_est": D_est,
        "flow_final": flow_final,
        "times_final": times_final,
        "obj_history": obj_history,
        "obj_ref": obj_ref,
        "obj_est": obj_est,
    }

    D_test = results["D_est"]
    # выбери пару крупных off-diagonal элементов
    idx = np.argwhere(~np.eye(D_test.shape[0], dtype=bool))
    vals = np.array([D_test[i,j] for i,j in idx])
    top = idx[np.argsort(-vals)[:5]]

    for i, j in top:
        fd_check_one(problem, D_test, int(i), int(j), rel_step=1e-6)

    save_objective_curve(obj_history, plot_path, "Mirror descent objective")

    return results


def run_levenberg_marquardt(
    csr,
    edge_cost,
    D_reference: np.ndarray,
    f_hat: np.ndarray,
    fw_eval_params: dict,
    *,
    mask: Optional[np.ndarray] = None,
    n_iters: int = 15,
    reg_kind: str = "none",          # "none" | "kl"
    reg_lambda: float = 0.0,
    D_prior: Optional[np.ndarray] = None,
    lm_lambda0: float = 1e-1,
    plot_path: Optional[Path] = None,
) -> dict:
    """
    Гаусс–Ньютон с демпфированием Левенберга–Марквардта по OD-матрице.
    Проекции на маргиналии оставляем те же, что и в зеркальном спуске.
    """
    D_reference = np.asarray(D_reference, dtype=np.float64)
    n = D_reference.shape[0]

    L_ref = D_reference.sum(axis=1)
    W_ref = D_reference.sum(axis=0)

    allowed = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(allowed, 0.0)

    if reg_kind == "kl" and reg_lambda > 0.0 and D_prior is None:
        D_prior = np.outer(L_ref, W_ref) / max(W_ref.sum(), 1e-12)
        D_prior = project_to_marginals_masked(D_prior, L_ref, W_ref, allowed, n_iters=80)

    problem = LSProblem(
        csr,
        edge_cost,
        f_hat,
        fw_eval_params,
        mask=mask,
        reg_kind=reg_kind,
        reg_lambda=reg_lambda,
        D_prior=D_prior,
        kl_eps=1e-12,
        enforce_zero_diag_in_reg=True,
    )

    D_est = np.outer(L_ref, W_ref) / max(W_ref.sum(), 1e-12)
    D_est = project_to_marginals_masked(D_est, L_ref, W_ref, allowed, n_iters=80)

    mu = lm_lambda0
    mu_up = 5.0
    mu_down = 0.5
    mu_min = 1e-8
    mu_max = 1e5

    obj_history = []
    best_obj = float("inf")

    for k in range(n_iters):
        residual, jac, _ = problem.residual_and_jacobian(D_est)
        obj = 0.5 * float(np.dot(residual, residual))

        reg_grad_flat = None
        if problem.reg_kind == "kl" and problem.reg_lambda > 0.0:
            reg_val, kl_grad = problem._kl_value_and_grad(D_est)
            obj += problem.reg_lambda * reg_val
            reg_grad_flat = problem.reg_lambda * kl_grad.reshape(-1)

        grad_flow_flat = jac.T @ residual
        grad_total = grad_flow_flat if reg_grad_flat is None else grad_flow_flat + reg_grad_flat
        grad_norm = float(np.linalg.norm(grad_total))

        JJt = jac @ jac.T
        mu_inv = 1.0 / mu
        B = JJt * mu_inv + np.eye(JJt.shape[0], dtype=np.float64)

        try:
            solve_r = np.linalg.solve(B, residual)
        except np.linalg.LinAlgError:
            solve_r = np.linalg.lstsq(B, residual, rcond=None)[0]

        delta_flow = -mu_inv * (jac.T @ solve_r)

        if reg_grad_flat is None:
            delta_flat = delta_flow
        else:
            reg_vec = mu_inv * reg_grad_flat
            try:
                solve_reg = np.linalg.solve(B, jac @ reg_vec)
            except np.linalg.LinAlgError:
                solve_reg = np.linalg.lstsq(B, jac @ reg_vec, rcond=None)[0]
            delta_reg = -reg_vec + mu_inv * (jac.T @ solve_reg)
            delta_flat = delta_flow + delta_reg

        delta_mat = delta_flat.reshape(D_est.shape)

        candidate = D_est + delta_mat
        candidate = np.maximum(candidate, 0.0)
        candidate = project_to_marginals_masked(candidate, L_ref, W_ref, allowed, n_iters=30)

        obj_cand = problem.value_only(candidate)

        accepted = obj_cand < obj
        if accepted:
            D_est = candidate
            obj = obj_cand
            mu = max(mu * mu_down, mu_min)
        else:
            mu = min(mu * mu_up, mu_max)

        obj_history.append(obj)
        best_obj = min(best_obj, obj)
        step_norm = float(np.linalg.norm(delta_flat, ord=np.inf))

        print(
            f"iter={k:03d} obj={obj:.6e} best={best_obj:.6e} "
            f"mu={mu:.3e} grad_norm={grad_norm:.3e} step_inf={step_norm:.3e} accepted={accepted}"
        )

    flow_final, _ = fw_beckmann(csr, edge_cost, D_est, **fw_eval_params)
    times_final = edge_cost(flow_final)

    obj_ref = problem.value_only(D_reference)
    obj_est = problem.value_only(D_est)

    results = {
        "D_est": D_est,
        "flow_final": flow_final,
        "times_final": times_final,
        "obj_history": obj_history,
        "obj_ref": obj_ref,
        "obj_est": obj_est,
    }

    D_test = results["D_est"]
    idx = np.argwhere(~np.eye(D_test.shape[0], dtype=bool))
    vals = np.array([D_test[i, j] for i, j in idx])
    top = idx[np.argsort(-vals)[:5]]

    for i, j in top:
        fd_check_one(problem, D_test, int(i), int(j), rel_step=1e-6)

    save_objective_curve(obj_history, plot_path, "Levenberg–Marquardt objective")

    return results

    


def print_solution(label: str, csr: CSRGraph, flow: np.ndarray, times: np.ndarray, obj_ref: float, obj_est: float):
    print(f"\n{label}")
    print("Результирующие потоки на рёбрах (tail -> head):")
    for idx, (u, v, f, t) in enumerate(zip(csr.tail, csr.head, flow, times)):
        print(f"{idx:03d}: {u}->{v}  flow={f:.4f}  time={t:.4f}")

    print("\nDiagnostics:")
    print(f"obj(D_reference) = {obj_ref:.6e}")
    print(f"obj(D_est)       = {obj_est:.6e}")
    print(f"rel diff         = {(obj_est - obj_ref) / max(obj_ref, 1e-12):.3e}")


def fd_check_one(problem: LSProblem, D: np.ndarray, i: int, j: int, rel_step: float = 1e-6) -> None:
    assert i != j
    D = np.asarray(D, dtype=np.float64)
    _, G = problem.evaluate(D)

    dij = float(D[i, j])
    if dij <= 1e-9:
        print("D[i,j] too small, pick another entry")
        return

    eps = rel_step * max(dij, 1.0)
    eps = min(eps, 0.1 * dij)  # чтобы D-eps не ушло в отрицательное

    Dp = D.copy(); Dp[i, j] += eps
    Dm = D.copy(); Dm[i, j] -= eps

    fp = problem.value_only(Dp)
    fm = problem.value_only(Dm)
    fd = (fp - fm) / (2.0 * eps)

    print(f"FD check at ({i},{j}): grad={G[i,j]:.6e}, fd={fd:.6e}, ratio={G[i,j]/(fd+1e-18):.3e}")


# ============================================================
# Incoming index (reverse CSR): head-sorted edge ids
# ============================================================
def build_incoming_index(csr) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает (first_in, in_eid) так, что входящие рёбра в v:
      edges = in_eid[first_in[v] : first_in[v+1]]
    """
    order = np.argsort(csr.head, kind="stable").astype(np.int32)
    head_sorted = csr.head[order]
    first_in = np.zeros(csr.n + 1, dtype=np.int32)
    np.add.at(first_in, head_sorted + 1, 1)
    first_in = np.cumsum(first_in, dtype=np.int32)
    return first_in, order


# ============================================================
# Soft AON (logit on near-shortest incoming edges)
# ============================================================
def _soft_aon_prepare_origin(
    csr,
    first_in: np.ndarray,
    in_eid: np.ndarray,
    weight: np.ndarray,
    origin: int,
    *,
    theta: float,
    delta_rel: float,
    delta_abs: float,
) -> dict:
    """
    Подготовка кэша для одного origin:
    - dist, pe из Дейкстры
    - order_asc, order_desc по dist
    - для каждого v: cand_edges, cand_tails, cand_probs (по incoming edges)
    """
    dist, pe = csr.dijkstra(weight, origin)
    finite = np.isfinite(dist)
    order_asc = np.argsort(dist, kind="stable")
    order_asc = order_asc[finite[order_asc]]
    order_desc = order_asc[::-1]

    cand_edges = [None] * csr.n
    cand_tails = [None] * csr.n
    cand_probs = [None] * csr.n

    o = int(origin)
    tiny = 1e-12

    for v in order_asc:
        v = int(v)
        if v == o:
            continue

        dv = dist[v]
        if not np.isfinite(dv):
            continue

        delta = delta_abs + delta_rel * float(dv)

        edges = in_eid[first_in[v] : first_in[v + 1]]
        e_list = []
        u_list = []
        w_list = []

        for e in edges:
            e = int(e)
            u = int(csr.tail[e])
            du = dist[u]
            if not np.isfinite(du):
                continue
            # гарантируем убывание dist, чтобы не было циклов
            if du >= dv - tiny:
                continue

            slack = float(du + weight[e] - dv)
            if slack < 0.0:
                slack = 0.0
            if slack <= delta:
                e_list.append(e)
                u_list.append(u)
                w_list.append(np.exp(-theta * slack))

        if not e_list:
            # fallback: строгое дерево кратчайших путей
            e = int(pe[v])
            if e >= 0:
                cand_edges[v] = np.array([e], dtype=np.int32)
                cand_tails[v] = np.array([int(csr.tail[e])], dtype=np.int32)
                cand_probs[v] = np.array([1.0], dtype=np.float64)
            else:
                cand_edges[v] = np.array([], dtype=np.int32)
                cand_tails[v] = np.array([], dtype=np.int32)
                cand_probs[v] = np.array([], dtype=np.float64)
        else:
            ww = np.asarray(w_list, dtype=np.float64)
            s = float(ww.sum())
            if s <= 0.0:
                p = np.full(len(ww), 1.0 / len(ww), dtype=np.float64)
            else:
                p = ww / s
            cand_edges[v] = np.asarray(e_list, dtype=np.int32)
            cand_tails[v] = np.asarray(u_list, dtype=np.int32)
            cand_probs[v] = p

    return {
        "origin": o,
        "dist": dist,
        "order_asc": order_asc.astype(np.int32),
        "order_desc": order_desc.astype(np.int32),
        "cand_edges": cand_edges,
        "cand_tails": cand_tails,
        "cand_probs": cand_probs,
    }


def soft_aon_assign_logit(
    csr,
    first_in: np.ndarray,
    in_eid: np.ndarray,
    weight: np.ndarray,
    D: np.ndarray,
    *,
    theta: float = 10.0,
    delta_rel: float = 0.02,
    delta_abs: float = 1e-3,
) -> Tuple[np.ndarray, float, list]:
    """
    Soft AON loading:
    y = expected edge flows under logit choice over near-shortest incoming edges.

    Returns:
      y         : (m,)
      total_cost: sum_e y[e] * weight[e]
      cache     : list of per-origin caches (для VJP/JT)
    """
    weight = np.asarray(weight, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    y = np.zeros(csr.m, dtype=np.float64)
    total_cost = 0.0

    caches = []

    for origin in range(csr.n):
        cache = _soft_aon_prepare_origin(
            csr, first_in, in_eid, weight, origin,
            theta=theta, delta_rel=delta_rel, delta_abs=delta_abs
        )
        caches.append(cache)

        mass = D[origin, :].astype(np.float64, copy=True)
        o = cache["origin"]

        for v in cache["order_desc"]:
            v = int(v)
            if v == o:
                continue
            mv = float(mass[v])
            if mv == 0.0:
                continue

            edges = cache["cand_edges"][v]
            tails = cache["cand_tails"][v]
            probs = cache["cand_probs"][v]

            if edges.size == 0:
                continue

            # распределяем массу на предков
            for e, u, p in zip(edges, tails, probs):
                f = mv * float(p)
                y[int(e)] += f
                mass[int(u)] += f
                total_cost += f * float(weight[int(e)])

    return y, total_cost, caches


def soft_aon_JT_logit(
    csr,
    caches: list,
    vec_m: np.ndarray,
) -> np.ndarray:
    """
    Возвращает (J_softAON)^T vec_m как матрицу grad_D формы (n,n),
    где J_softAON: D -> y.

    Важно: это JT только для этапа загрузки (AON), при фиксированных weight/кэшах.
    """
    vec_m = np.asarray(vec_m, dtype=np.float64)
    n = int(csr.n)
    grad_D = np.zeros((n, n), dtype=np.float64)

    for cache in caches:
        o = int(cache["origin"])
        order_asc = cache["order_asc"]

        # direct[v] = sum_{e in cand(v)} vec[e] * p(e|v)
        direct = np.zeros(n, dtype=np.float64)
        for v in order_asc:
            v = int(v)
            if v == o:
                continue
            edges = cache["cand_edges"][v]
            probs = cache["cand_probs"][v]
            if edges is None or edges.size == 0:
                continue
            direct[v] = float(np.dot(vec_m[edges], probs))

        # DP: a[v] = direct[v] + sum_{e=u->v} p(e|v) * a[u]
        a = np.zeros(n, dtype=np.float64)
        for v in order_asc:
            v = int(v)
            if v == o:
                continue
            edges = cache["cand_edges"][v]
            tails = cache["cand_tails"][v]
            probs = cache["cand_probs"][v]
            if edges is None or edges.size == 0:
                continue
            acc = direct[v]
            for u, p in zip(tails, probs):
                acc += float(p) * a[int(u)]
            a[v] = acc

        grad_D[o, :] = a

    # диагональ OD не используем
    np.fill_diagonal(grad_D, 0.0)
    return grad_D


# ============================================================
# Soft Frank-Wolfe (straight-through JT): ignores dP/dD, ddist/dD
# ============================================================
def fw_beckmann_soft(
    csr,
    edge_cost,
    D: np.ndarray,
    *,
    max_iter: int = 30,
    rgap_target: float = 1e-3,
    theta: float = 10.0,
    delta_rel: float = 0.02,
    delta_abs: float = 1e-3,
    verbose: bool = False,
) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """
    Возвращает:
      flow : (m,) финальные потоки
      JT   : функция, которая по vec_m (m,) возвращает приближённый grad_D (n,n) = J^T vec_m

    JT здесь "straight-through":
      - учитывает только линейную зависимость soft assignment от D при фиксированных весах на каждой итерации
      - не дифференцирует через обновление весов, и не дифференцирует dist в Dijkstra
    Это ровно то, что обычно ломается на жёстком AON, а soft AON делает заметно стабильнее.
    """
    D = np.asarray(D, dtype=np.float64)

    first_in, in_eid = build_incoming_index(csr)

    flow = np.zeros(csr.m, dtype=np.float64)
    weights_list = []
    gammas_list = []
    caches_list = []

    for k in range(1, max_iter + 1):
        w = np.asarray(edge_cost(flow), dtype=np.float64)
        y, _, caches = soft_aon_assign_logit(
            csr, first_in, in_eid, w, D,
            theta=theta, delta_rel=delta_rel, delta_abs=delta_abs
        )

        gamma = 2.0 / (k + 2.0)
        flow = (1.0 - gamma) * flow + gamma * y

        weights_list.append(w)
        gammas_list.append(gamma)
        caches_list.append(caches)

        if verbose and (k == 1 or k % 10 == 0):
            print(f"iter={k:4d} gamma={gamma:.6f}")

    def JT(vec_m: np.ndarray) -> np.ndarray:
        g = np.asarray(vec_m, dtype=np.float64)
        grad_D = np.zeros((csr.n, csr.n), dtype=np.float64)

        # reverse through convex combinations (straight-through)
        for caches, gamma in zip(reversed(caches_list), reversed(gammas_list)):
            grad_D += soft_aon_JT_logit(csr, caches, gamma * g)
            g = (1.0 - gamma) * g

        np.fill_diagonal(grad_D, 0.0)
        return grad_D

    return flow, JT


# ============================================================
# main4: experiment (soft FW + KL + mirror descent)
# ============================================================
def main4(
    OD_PATH: Path,
    load_od_matrix,
    build_dense_graph,
    project_to_marginals_masked,
    *,
    observed_fraction: float = 0.10,
    theta: float = 10.0,
    reg_lambda: float = 1e-3,
    n_iters_outer: int = 30,
    max_iter_fw: int = 30,
):
    D_reference = load_od_matrix(OD_PATH)
    n = D_reference.shape[0]
    L_ref = D_reference.sum(axis=1)
    W_ref = D_reference.sum(axis=0)

    allowed = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(allowed, 0.0)

    csr, edge_cost = build_dense_graph(n)

    # "наблюдения": берём soft FW на референсе (можно заменить на ваш hard FW)
    flow_ref, _ = fw_beckmann_soft(
        csr, edge_cost, D_reference,
        max_iter=max_iter_fw,
        theta=theta,
        delta_rel=0.02,
        delta_abs=1e-3,
        verbose=False,
    )

    rng = np.random.default_rng(123)
    mask = (rng.random(flow_ref.shape) < observed_fraction).astype(np.float64)
    if mask.sum() == 0:
        mask[0] = 1.0
    print(f"main4: observed_fraction={mask.mean():.0%}, theta={theta}, reg_lambda={reg_lambda}")

    f_hat = flow_ref.copy()

    # prior для KL: IPF с теми же маргиналиями
    D_prior = np.outer(L_ref, W_ref) / max(W_ref.sum(), 1e-12)
    D_prior = project_to_marginals_masked(D_prior, L_ref, W_ref, allowed, n_iters=80)

    # старт
    D_est = D_prior.copy()

    def kl_value_and_grad(D: np.ndarray, eps: float = 1e-12) -> Tuple[float, np.ndarray]:
        Dp = np.maximum(D, eps)
        Pp = np.maximum(D_prior, eps)
        log_ratio = np.log(Dp) - np.log(Pp)
        kl_mat = Dp * log_ratio - Dp + Pp
        kl_val = float(np.sum(allowed * kl_mat))
        kl_grad = allowed * log_ratio
        np.fill_diagonal(kl_grad, 0.0)
        return kl_val, kl_grad

    # mirror descent outer
    step0 = 1e-2
    ls_beta = 0.5
    ls_min = 1e-12
    ls_max_trials = 20
    improve_eps = 1e-12
    eps_floor = 1e-12
    exp_clip = 50.0

    obj_history = []

    for it in range(n_iters_outer):
        flow, JT = fw_beckmann_soft(
            csr, edge_cost, D_est,
            max_iter=max_iter_fw,
            theta=theta,
            delta_rel=0.02,
            delta_abs=1e-3,
            verbose=False,
        )

        residual = mask * (flow - f_hat)
        data = 0.5 * float(np.dot(residual, residual))

        kl_val, kl_grad = kl_value_and_grad(D_est)
        obj = data + reg_lambda * kl_val

        grad_data = JT(residual)
        grad = grad_data + reg_lambda * kl_grad

        grad_norm = float(np.linalg.norm(grad))

        step = step0
        accepted = False
        best_obj = obj
        best_D = D_est

        for _ in range(ls_max_trials):
            base = np.maximum(D_est, eps_floor) * allowed
            expo = np.clip(-step * grad, -exp_clip, exp_clip)
            cand = base * np.exp(expo)

            cand = project_to_marginals_masked(cand, L_ref, W_ref, allowed, n_iters=50)

            flow_c, _ = fw_beckmann_soft(
                csr, edge_cost, cand,
                max_iter=max_iter_fw,
                theta=theta,
                delta_rel=0.02,
                delta_abs=1e-3,
                verbose=False,
            )
            res_c = mask * (flow_c - f_hat)
            data_c = 0.5 * float(np.dot(res_c, res_c))
            kl_c, _ = kl_value_and_grad(cand)
            obj_c = data_c + reg_lambda * kl_c

            if obj_c < best_obj - improve_eps:
                best_obj = obj_c
                best_D = cand
                accepted = True
                break

            step *= ls_beta
            if step < ls_min:
                break

        D_est = best_D

        if accepted:
            step0 = min(step0 * 1.2, 1e-1)
        else:
            step0 *= 0.5

        obj_history.append(best_obj)

        print(
            f"outer={it:03d} obj={obj:.6e} best={best_obj:.6e} "
            f"data={data:.3e} kl={kl_val:.3e} grad_norm={grad_norm:.3e} "
            f"step={step:.2e} accepted={accepted}"
        )

    # финальная диагностика
    flow_final, _ = fw_beckmann_soft(
        csr, edge_cost, D_est,
        max_iter=max_iter_fw,
        theta=theta,
        delta_rel=0.02,
        delta_abs=1e-3,
        verbose=False,
    )
    save_objective_curve(obj_history, Path("objective_curve_main4.png"), "Soft FW + KL mirror objective")

    res_f = mask * (flow_final - f_hat)
    data_f = 0.5 * float(np.dot(res_f, res_f))
    kl_f, _ = kl_value_and_grad(D_est)
    print("\nmain4 final:")
    print("data =", data_f, "kl =", kl_f, "total =", data_f + reg_lambda * kl_f)
    print("diag mass =", float(np.sum(np.diag(D_est))))


def main1():
    """
    Проверка восстановления OD-матрицы при наличии шума в потоках.
    """
    D_reference = load_od_matrix(OD_PATH)

    csr, edge_cost = build_dense_graph(D_reference.shape[0])
    fw_eval_params = {"max_iter": 30, "rgap_target": 1e-3, "verbose": False, "use_numba": True}
    flow_ref, _ = fw_beckmann(csr, edge_cost, D_reference, **fw_eval_params)

    rng = np.random.default_rng(42)
    noise = 0.05 * np.maximum(np.abs(flow_ref), 1.0) * rng.standard_normal(flow_ref.shape)
    f_hat = flow_ref + noise

    results = run_mirror_descent(
        csr,
        edge_cost,
        D_reference,
        f_hat,
        fw_eval_params,
        plot_path=Path("objective_curve_main1.png"),
    )

    print_solution("Сценарий main1: шумленные потоки", csr, results["flow_final"], results["times_final"], results["obj_ref"], results["obj_est"])


def main2():
    """
    Частично наблюдаемые потоки: оптимизируем по маске.
    Добавляем KL-регуляризацию к prior.
    """
    D_reference = load_od_matrix(OD_PATH)

    csr, edge_cost = build_dense_graph(D_reference.shape[0])
    fw_eval_params = {"max_iter": 30, "rgap_target": 1e-3, "verbose": False, "use_numba": True}
    flow_ref, _ = fw_beckmann(csr, edge_cost, D_reference, **fw_eval_params)

    rng = np.random.default_rng(123)
    observed_fraction = 0.10
    mask = (rng.random(flow_ref.shape) < observed_fraction).astype(np.float64)
    if mask.sum() == 0:
        mask[0] = 1.0
    print(f"Доля наблюдаемых потоков: {mask.mean():.0%}")

    # KL-regularization strength: подбирать.
    # Стартовые варианты: 1e-4, 1e-3, 1e-2 (в зависимости от масштабов D).
    reg_lambda = 1e-3

    results = run_mirror_descent(
        csr,
        edge_cost,
        D_reference,
        flow_ref,                 # f_hat = flow_ref (без шума)
        fw_eval_params,
        mask=mask,
        n_iters=30,
        plot_path=Path("objective_curve_main2.png"),
        reg_kind="kl",
        reg_lambda=reg_lambda,
        D_prior=None             # возьмётся IPF-prior внутри
    )

    print_solution(
        f"Сценарий main2: маска + KL-reg (lambda={reg_lambda:g})",
        csr,
        results["flow_final"],
        results["times_final"],
        results["obj_ref"],
        results["obj_est"],
    )


def main3():
    """
    Аналог main1, но оптимизация OD-матрицы через Левенберга–Марквардта.
    """
    D_reference = load_od_matrix(OD_PATH)

    csr, edge_cost = build_dense_graph(D_reference.shape[0])
    fw_eval_params = {"max_iter": 30, "rgap_target": 1e-3, "verbose": False, "use_numba": True}
    flow_ref, _ = fw_beckmann(csr, edge_cost, D_reference, **fw_eval_params)

    rng = np.random.default_rng(42)
    noise = 0.05 * np.maximum(np.abs(flow_ref), 1.0) * rng.standard_normal(flow_ref.shape)
    f_hat = flow_ref + noise

    results = run_levenberg_marquardt(
        csr,
        edge_cost,
        D_reference,
        f_hat,
        fw_eval_params,
        n_iters=15,
        plot_path=Path("objective_curve_main3.png"),
    )

    print_solution(
        "Сценарий main3: шумленные потоки (LM)",
        csr,
        results["flow_final"],
        results["times_final"],
        results["obj_ref"],
        results["obj_est"],
    )


if __name__ == "__main__":
    main4(
        OD_PATH=OD_PATH,
        load_od_matrix=load_od_matrix,
        build_dense_graph=build_dense_graph,
        project_to_marginals_masked=project_to_marginals_masked,
        observed_fraction=0.10,
        theta=10.0,
        reg_lambda=1e-3,
        n_iters_outer=20,
        max_iter_fw=20,
    )
