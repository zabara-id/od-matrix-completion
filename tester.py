import csv
from pathlib import Path
from typing import Optional, Tuple

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
    0.5 * || M (f(D) - f_hat) ||^2_2 + 0.5 * reg_lambda * ||D - D_prior||_F^2
    """

    def __init__(
        self,
        csr: CSRGraph,
        edge_cost: BRP,
        f_hat: np.ndarray,
        fw_kwargs: dict,
        mask: Optional[np.ndarray] = None,
        *,
        reg_lambda: float = 0.0,
        D_prior: Optional[np.ndarray] = None,
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

        # regularization
        self.reg_lambda = float(reg_lambda)
        if self.reg_lambda < 0.0:
            raise ValueError("reg_lambda must be >= 0")

        if self.reg_lambda > 0.0:
            if D_prior is None:
                raise ValueError("D_prior must be provided when reg_lambda > 0")
            self.D_prior = np.asarray(D_prior, dtype=np.float64)
        else:
            self.D_prior = None

    def _apply_mask(self, residual: np.ndarray) -> np.ndarray:
        if self.mask is None:
            return residual
        return residual * self.mask

    def evaluate(self, D: np.ndarray) -> Tuple[float, np.ndarray]:
        D = np.asarray(D, dtype=np.float64)

        flow, grad = fw_beckmann(self.graph, self.edge_cost, D, **self.fw_kwargs)
        residual = self._apply_mask(flow - self.f_hat)

        value = 0.5 * float(np.dot(residual, residual))

        grad_D = (grad.T @ residual).reshape(D.shape)

        if self.reg_lambda > 0.0:
            diff = D - self.D_prior
            value += 0.5 * self.reg_lambda * float(np.sum(diff * diff))
            grad_D += self.reg_lambda * diff

        return value, grad_D

    def value_only(self, D: np.ndarray) -> float:
        D = np.asarray(D, dtype=np.float64)

        flow, _ = fw_beckmann(self.graph, self.edge_cost, D, **self.fw_kwargs)
        residual = self._apply_mask(flow - self.f_hat)

        value = 0.5 * float(np.dot(residual, residual))

        if self.reg_lambda > 0.0:
            diff = D - self.D_prior
            value += 0.5 * self.reg_lambda * float(np.sum(diff * diff))

        return value



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


def run_mirror_descent(
    csr: CSRGraph,
    edge_cost: BRP,
    D_reference: np.ndarray,
    f_hat: np.ndarray,
    fw_eval_params: dict,
    *,
    mask: Optional[np.ndarray] = None,
    n_iters: int = 30,
    plot_path: Optional[Path] = None,
) -> dict:
    """
    Оптимизация OD-матрицы по наблюдаемым потокам f_hat (с маской или без).
    Возвращает финальные значения и историю функции цели.
    """
    L_ref = D_reference.sum(axis=1)
    W_ref = D_reference.sum(axis=0)
    problem = LSProblem(csr, edge_cost, f_hat, fw_eval_params, mask=mask)

    # Старт: IPF-аппроксимация вместо D_reference, чтобы видеть динамику алгоритма
    D_est = np.outer(L_ref, W_ref) / max(W_ref.sum(), 1e-12)
    np.fill_diagonal(D_est, 0.0)
    D_est = project_to_marginals(D_est, L_ref, W_ref, n_iters=10)

    obj_history = []

    # Mirror Descent параметры
    step0 = 1e-2
    ls_beta = 0.5
    ls_min = 1e-12
    ls_max_trials = 20
    improve_eps = 1e-12

    eps_floor = 1e-12  # чтоб не "залипать" в нулях при мультипликативном обновлении
    exp_clip = 50.0    # защита от overflow в exp

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

            base = np.maximum(D_est, eps_floor)
            expo = np.clip(-step * subgrad_D, -exp_clip, exp_clip)
            candidate = base * np.exp(expo)

            np.fill_diagonal(candidate, 0.0)
            candidate = project_to_marginals(candidate, L_ref, W_ref, n_iters=5)

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

    if plot_path:
        plt.figure(figsize=(6, 3))
        plt.plot(obj_history, marker="o", linewidth=1)
        plt.xlabel("Итерация")
        plt.ylabel("0.5 * ||f(D)-f_hat||^2")
        plt.title("Сходимость mirror descent по D")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        print(f"\nГрафик сохранён в {plot_path}")

    return {
        "D_est": D_est,
        "flow_final": flow_final,
        "times_final": times_final,
        "obj_history": obj_history,
        "obj_ref": obj_ref,
        "obj_est": obj_est,
    }


def print_solution(label: str, csr: CSRGraph, flow: np.ndarray, times: np.ndarray, obj_ref: float, obj_est: float):
    print(f"\n{label}")
    print("Результирующие потоки на рёбрах (tail -> head):")
    for idx, (u, v, f, t) in enumerate(zip(csr.tail, csr.head, flow, times)):
        print(f"{idx:03d}: {u}->{v}  flow={f:.4f}  time={t:.4f}")

    print("\nDiagnostics:")
    print(f"obj(D_reference) = {obj_ref:.6e}")
    print(f"obj(D_est)       = {obj_est:.6e}")
    print(f"rel diff         = {(obj_est - obj_ref) / max(obj_ref, 1e-12):.3e}")


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
    Проверка, когда известна только часть потоков: оцениваем по маске наблюдений без шума.
    """
    D_reference = load_od_matrix(OD_PATH)

    csr, edge_cost = build_dense_graph(D_reference.shape[0])
    fw_eval_params = {"max_iter": 30, "rgap_target": 1e-3, "verbose": False, "use_numba": True}
    flow_ref, _ = fw_beckmann(csr, edge_cost, D_reference, **fw_eval_params)

    rng = np.random.default_rng(123)
    observed_fraction = 0.35
    mask = (rng.random(flow_ref.shape) < observed_fraction).astype(np.float64)
    if mask.sum() == 0:
        mask[0] = 1.0
    print(f"Доля наблюдаемых потоков: {mask.mean():.0%}")

    results = run_mirror_descent(
        csr,
        edge_cost,
        D_reference,
        flow_ref,
        fw_eval_params,
        mask=mask,
        plot_path=Path("objective_curve_main2.png"),
    )

    print_solution(
        "Сценарий main2: частично наблюдаемые потоки (маска)",
        csr,
        results["flow_final"],
        results["times_final"],
        results["obj_ref"],
        results["obj_est"],
    )


if __name__ == "__main__":
    main1()
