import csv
import os
from pathlib import Path
from typing import Dict

import numpy as np

from mirror_descent_completion_kl import mirror_descent_completion
from optimize_results_dto import MirrorDescentResult
from src.od_matrix_completion.core.models.manyalli_written_beckmann import (
    BRP,
    CSRGraph,
    fw_beckmann_flow,
)

# Matplotlib config/cache dir: keep it in-repo to avoid permission issues on macOS.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


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


def _get_plt():
    import sys
    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def load_od_matrix(path: str) -> np.ndarray:
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
            plt.semilogy(x, y, marker="o", ms=2, label=label)
        else:
            plt.plot(x, y, marker="o", ms=2, label=label)

    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
