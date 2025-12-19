import csv
import os
from pathlib import Path
from typing import Dict

import numpy as np

from mirror_descent_beckmann import mirror_descent_beckmann
from optimize_results_dto import MirrorDescentResult
from src.od_matrix_completion.core.models.manyalli_written_beckmann import (
    BRP,
    CSRGraph,
    fw_beckmann_flow,
)

OD_PATH = Path("data/processed/Mat_Car_ev.csv")

# Matplotlib config/cache dir: keep it in-repo to avoid permission issues on macOS.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def _get_plt():
    import sys

    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def load_od_matrix(path: Path = OD_PATH) -> np.ndarray:
    """
    Читает матрицу корреспонденций из TSV и возвращает np.ndarray с нулевой диагональю.
    """
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
            matrix_rows.append([float(cell) for cell in row[1:]])
        except ValueError as exc:
            raise ValueError(f"Не удалось распарсить числа в строке {idx}") from exc

    mat = np.asarray(matrix_rows, dtype=np.float64)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"OD-матрица должна быть квадратной, получено {mat.shape}")

    np.fill_diagonal(mat, 0.0)
    return mat


def build_dense_graph(n_nodes: int) -> tuple[CSRGraph, BRP]:
    """
    Строит связный ориентированный граф без петель с 4·n рёбрами.
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
        if u == v or (u, v) in edge_set:
            return
        edge_set.add((u, v))
        edges.append((u, v))

    for i in range(n_nodes):
        j = (i + 1) % n_nodes
        add_edge(i, j)
        add_edge(j, i)

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

    if len(edges) < target_edges:
        for u in range(n_nodes):
            for v in range(n_nodes):
                if len(edges) >= target_edges:
                    break
                add_edge(u, v)

    edges = edges[:target_edges]
    tail = np.fromiter((u for u, _ in edges), dtype=np.int32, count=target_edges)
    head = np.fromiter((v for _, v in edges), dtype=np.int32, count=target_edges)
    csr = CSRGraph.from_edges(n_nodes, tail, head)

    cap = 1800.0 + 200.0 * ((tail + head) % 4)
    t0 = 1.0 + 0.1 * ((np.abs(head - tail) % 5))
    alpha = np.full(csr.m, 0.15, dtype=np.float64)
    beta = np.full(csr.m, 4.0, dtype=np.float64)
    edge_cost = BRP(cap, t0, alpha, beta)

    return csr, edge_cost


def plot_history(series: Dict[str, list[float]], path: Path, title: str, ylabel: str, *, semilogy: bool = True):
    plt = _get_plt()
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    for label, values in series.items():
        y = np.asarray(values, dtype=np.float64)
        x = np.arange(y.size)
        if semilogy:
            plt.semilogy(x, y, marker="o", ms=3, label=label)
        else:
            plt.plot(x, y, marker="o", ms=3, label=label)

    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_demo(
    observed_fraction: float = 0.15,
    reference_noise_level: float = 0.005,
    mask_seed: int = 42,
    exact_mask: bool = True,
) -> Dict[str, MirrorDescentResult]:
    D_reference_true = load_od_matrix(OD_PATH)
    rng = np.random.default_rng(123)
    ref_noise = reference_noise_level * rng.standard_normal(D_reference_true.shape)
    D_reference = D_reference_true * (1.0 + ref_noise)
    D_reference = np.maximum(D_reference, 0.0)
    np.fill_diagonal(D_reference, 0.0)

    csr, edge_cost = build_dense_graph(D_reference.shape[0])

    fw_hard_kwargs = {"max_iter": 30, "rgap_target": 1e-3, "verbose": False, "use_numba": True}
    flow_ref = fw_beckmann_flow(csr, edge_cost, D_reference_true, **fw_hard_kwargs)

    if exact_mask:
        m = flow_ref.size
        k = max(1, min(m, int(round(observed_fraction * m))))
        rng_mask = np.random.default_rng(mask_seed)
        perm = rng_mask.permutation(m)
        mask = np.zeros(m, dtype=np.float64)
        mask[perm[:k]] = 1.0
    else:
        rng_mask = np.random.default_rng(mask_seed)
        mask = (rng_mask.random(flow_ref.shape) < observed_fraction).astype(np.float64)
        if mask.sum() == 0:
            mask[0] = 1.0
    f_hat = flow_ref.copy()

    fw_soft_kwargs = {
        "max_iter": 30,
        "theta": 10.0,
        "delta_rel": 0.02,
        "delta_abs": 1e-3,
        "verbose": False,
        "use_numba": True,
    }

    def make_progress_printer(mode: str, total: int):
        bar_len = 30

        def _cb(it: int, total_iters: int, _: str):
            done = it + 1
            filled = int(bar_len * done / total_iters)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(f"\r[{mode}] |{bar}| {done}/{total_iters}", end="", flush=True)

        return _cb

    def make_switch_logger(mode: str):
        def _on_switch(iter_idx: int):
            print(f"\n[{mode}] переключаюсь на soft-град на итерации {iter_idx}")
        return _on_switch

    n_iters = 50
    results: Dict[str, MirrorDescentResult] = {}
    for mode in ("hard", "soft_grad", "auto_soft"):
        print(f"\nЗапускаю зеркальный спуск в режиме '{mode}' (observed {mask.mean():.0%} потоков)")
        progress_cb = make_progress_printer(mode, total=n_iters)
        switch_cb = make_switch_logger(mode) if mode == "auto_soft" else None
        res = mirror_descent_beckmann(
            csr,
            edge_cost,
            f_hat,
            D_reference=D_reference,
            mode=mode,
            mask=mask,
            reg_lambda=1e-3,
            n_iters=n_iters,
            fw_hard_kwargs=fw_hard_kwargs,
            fw_soft_kwargs=fw_soft_kwargs,
            progress_callback=progress_cb,
            switch_callback=switch_cb,
        )
        print()  # завершить строку прогресса
        results[mode] = res

        final_obj = res.objective_history[-1]
        final_rel = res.rel_l1_history[-1]
        switch_msg = f", switch_iter={res.switch_iter}" if res.switch_iter is not None else ""
        print(f"  final objective={final_obj:.6e}, rel_l1={final_rel:.3e}{switch_msg}")

    return results


def main():
    observed_fraction = 0.20  # доля известных потоков
    results = run_demo(
        observed_fraction=observed_fraction,
        reference_noise_level=0.005,
        mask_seed=42,
        exact_mask=True,
    )

    plot_history(
        {name: res.objective_history for name, res in results.items()},
        Path("plots_part2/objective_curves.png"),
        f"Hard objective (observed {observed_fraction:.0%} flows)",
        "objective",
        semilogy=True,
    )
    plot_history(
        {name: res.rel_l1_history for name, res in results.items()},
        Path("plots_part2/rel_l1_curves.png"),
        f"||D_k - D_ref||_1 / ||D_ref||_1 (observed {observed_fraction:.0%})",
        "relative L1 error",
        semilogy=True,
    )

    print("\nГрафики сохранены в папке plots_part/.")


if __name__ == "__main__":
    main()
