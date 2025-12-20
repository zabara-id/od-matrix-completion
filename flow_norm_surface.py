from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

# Keep matplotlib cache in-repo (macOS permission friendly).
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from subfunctions import build_dense_graph, load_od_matrix
from src.od_matrix_completion.core.models.manyalli_written_beckmann import (
    CSRGraph,
    BRP,
    aon_assign_flow,
    stop_criterion,
)


NormKind = Literal["l2", "l1", "linf"]
ProjectionKind = Literal["cycles", "pairs"]

# ==========================
# Настройки (правь тут)
# ==========================
OD_PATH = Path("data/processed/Mat_Car_ev.csv")

# Какую 2D-проекцию в OD-пространстве смотреть:
# - "cycles": два независимых 2x2 cycle move (сохраняет маргиналии)
# - "pairs" : две конкретные OD-пары (маргиналии не сохраняет)
PROJECTION: ProjectionKind = "cycles"

# Сетка по (a,b): GRID x GRID запусков Beckmann
GRID = 21

# Амплитуда (относительная) для 2D-бокса вокруг D0. Меньше = быстрее/стабильнее.
REL = 0.25

# Норма потока по рёбрам ||f(D)||
NORM: NormKind = "l2"

# Если True: рисуем ||f(D) - f(D0)|| вместо ||f(D)||
DELTA = False

# Для скорости можно брать только первые N зон (<= полного размера)
N_ZONES = 25

# Подмножество зон: "first" или "random"
SUBSET: Literal["first", "random"] = "first"
SUBSET_SEED = 123

# Настройки FW внутри Beckmann
FW_MAX_ITER = 20
FW_RGAP = 1e-3
USE_NUMBA = True
WARM_START = True

# Заголовок (пусто => автогенерация)
TITLE = ""

# Опционально: сохранить сетку/картинку (оставь None если не нужно)
SAVE_NPZ: Path | None = None  # например: Path("plots/flow_norm_surface.npz")
OUT_PNG: Path | None = None   # например: Path("plots/flow_norm_surface.png")


def flow_norm(flow: np.ndarray, kind: NormKind) -> float:
    flow = np.asarray(flow, dtype=np.float64)
    if kind == "l2":
        return float(np.linalg.norm(flow))
    if kind == "l1":
        return float(np.sum(np.abs(flow)))
    if kind == "linf":
        return float(np.max(np.abs(flow)))
    raise ValueError(f"Unknown norm kind: {kind}")


def fw_beckmann_flow_warm(
    csr: CSRGraph,
    edge_cost: BRP,
    D: np.ndarray,
    *,
    flow0: np.ndarray | None,
    max_iter: int,
    rgap_target: float,
    use_numba: bool,
) -> np.ndarray:
    """
    Same as fw_beckmann_flow, but optionally warm-starts from flow0.
    """
    if flow0 is None:
        flow = np.zeros(csr.m, dtype=np.float64)
    else:
        flow = np.asarray(flow0, dtype=np.float64).copy()

    for k in range(1, int(max_iter) + 1):
        weight = edge_cost(flow)
        y, total_cost_k = aon_assign_flow(csr, weight, D, use_numba=use_numba)

        gamma = 2.0 / (k + 2.0)
        flow = (1.0 - gamma) * flow + gamma * y

        rg = stop_criterion(flow, edge_cost(flow), total_cost_k)
        if rg <= float(rgap_target):
            break

    return flow


@dataclass(frozen=True)
class SparseDirection:
    # coords[k] is (i, j), coeffs[k] is the coefficient at that entry.
    coords: tuple[tuple[int, int], ...]
    coeffs: tuple[float, ...]

    def support(self) -> set[tuple[int, int]]:
        return set(self.coords)


def _pick_cycle_direction(
    D0: np.ndarray,
    rng: np.random.Generator,
    *,
    forbidden_cells: set[tuple[int, int]],
    min_value: float = 1e-6,
    max_tries: int = 50_000,
) -> SparseDirection:
    """
    Picks a 2x2 "cycle move" that preserves row/col sums:
      +1 at (o1,d1) and (o2,d2)
      -1 at (o1,d2) and (o2,d1)
    """
    n = int(D0.shape[0])
    for _ in range(int(max_tries)):
        o1, o2 = rng.choice(n, size=2, replace=False).tolist()
        d1, d2 = rng.choice(n, size=2, replace=False).tolist()

        # Avoid touching the diagonal.
        if o1 in (d1, d2) or o2 in (d1, d2):
            continue

        cells = ((o1, d1), (o2, d2), (o1, d2), (o2, d1))
        if any(c in forbidden_cells for c in cells):
            continue

        values = [float(D0[i, j]) for i, j in cells]
        if min(values) <= float(min_value):
            continue

        return SparseDirection(coords=cells, coeffs=(+1.0, +1.0, -1.0, -1.0))

    raise RuntimeError("Failed to sample a valid cycle direction (try different seed/subset).")


def _top_od_pairs(D0: np.ndarray, k: int = 2) -> list[tuple[int, int]]:
    n = int(D0.shape[0])
    off = ~np.eye(n, dtype=bool)
    flat_idx = np.argsort(-D0[off].reshape(-1))
    coords = np.argwhere(off)
    chosen = [tuple(map(int, coords[i])) for i in flat_idx[: int(k)]]
    return chosen


def _apply_sparse(D: np.ndarray, direction: SparseDirection, scale: float) -> None:
    for (i, j), c in zip(direction.coords, direction.coeffs):
        D[int(i), int(j)] += float(scale) * float(c)


def _safe_scale_for_box(
    D0: np.ndarray,
    dir_u: SparseDirection,
    dir_v: SparseDirection,
    *,
    rel: float,
) -> float:
    """
    We will explore (a,b) in [-s, s] x [-s, s] (same scale for both).
    Ensures D0 + a U + b V stays nonnegative on the union support for all (a,b) in the box.
    """
    rel = float(rel)
    if rel <= 0.0:
        raise ValueError("rel must be > 0")

    # For each affected cell: |a|*|u| + |b|*|v| <= s*(|u|+|v|) must be <= D0_cell.
    combined: dict[tuple[int, int], tuple[float, float]] = {}
    for (i, j), c in zip(dir_u.coords, dir_u.coeffs):
        combined[(int(i), int(j))] = (float(c), 0.0)
    for (i, j), c in zip(dir_v.coords, dir_v.coeffs):
        key = (int(i), int(j))
        u, _ = combined.get(key, (0.0, 0.0))
        combined[key] = (u, float(c))

    s_max = float("inf")
    for (i, j), (cu, cv) in combined.items():
        denom = abs(float(cu)) + abs(float(cv))
        if denom <= 0.0:
            continue
        s_max = min(s_max, float(D0[i, j]) / denom)

    if not np.isfinite(s_max) or s_max <= 0.0:
        raise RuntimeError("Could not compute a positive safe scale.")

    return rel * s_max


def main() -> None:
    import matplotlib.pyplot as plt

    D_full = load_od_matrix(OD_PATH)
    n_full = int(D_full.shape[0])

    n = min(int(N_ZONES), n_full)
    if n < 3:
        raise ValueError("N_ZONES must be >= 3")

    if SUBSET == "first":
        idx = np.arange(n, dtype=int)
    else:
        rng = np.random.default_rng(int(SUBSET_SEED))
        idx = np.sort(rng.choice(n_full, size=n, replace=False))

    D0 = np.asarray(D_full[np.ix_(idx, idx)], dtype=np.float64).copy()
    np.fill_diagonal(D0, 0.0)

    csr, edge_cost = build_dense_graph(n)

    rng = np.random.default_rng(int(SUBSET_SEED))
    projection: ProjectionKind = PROJECTION
    norm_kind: NormKind = NORM

    if projection == "pairs":
        (i1, j1), (i2, j2) = _top_od_pairs(D0, k=2)
        scale1 = float(REL) * float(D0[i1, j1])
        scale2 = float(REL) * float(D0[i2, j2])
        a_vals = np.linspace(-scale1, scale1, int(GRID))
        b_vals = np.linspace(-scale2, scale2, int(GRID))
        dir_u = SparseDirection(coords=((i1, j1),), coeffs=(1.0,))
        dir_v = SparseDirection(coords=((i2, j2),), coeffs=(1.0,))
        meta = f"pairs: ({i1},{j1}) and ({i2},{j2}), scales=({scale1:.3g},{scale2:.3g})"
    else:
        # Two independent cycle moves (they preserve marginals).
        d1 = _pick_cycle_direction(D0, rng, forbidden_cells=set())
        d2 = _pick_cycle_direction(D0, rng, forbidden_cells=d1.support())
        scale = _safe_scale_for_box(D0, d1, d2, rel=float(REL))
        a_vals = np.linspace(-scale, scale, int(GRID))
        b_vals = np.linspace(-scale, scale, int(GRID))
        dir_u, dir_v = d1, d2
        meta = f"cycles: U={d1.coords}, V={d2.coords}, scale={scale:.3g}"

    print(f"OD size n={n} (from n_full={n_full}), edges m={csr.m}")
    print(f"projection={projection}, norm={norm_kind}, delta={bool(DELTA)}")
    print(f"FW: max_iter={FW_MAX_ITER}, rgap={FW_RGAP:g}, numba={bool(USE_NUMBA)}, warm_start={bool(WARM_START)}")
    print(meta)

    flow_base = fw_beckmann_flow_warm(
        csr,
        edge_cost,
        D0,
        flow0=None,
        max_iter=int(FW_MAX_ITER),
        rgap_target=float(FW_RGAP),
        use_numba=bool(USE_NUMBA),
    )

    Z = np.zeros((a_vals.size, b_vals.size), dtype=np.float64)

    last_flow: np.ndarray | None = flow_base if WARM_START else None

    total = int(a_vals.size * b_vals.size)
    done = 0

    for ia, a in enumerate(a_vals):
        # Row-wise warm start.
        row_flow0 = last_flow if WARM_START else None
        for ib, b in enumerate(b_vals):
            Dc = D0.copy()
            _apply_sparse(Dc, dir_u, float(a))
            _apply_sparse(Dc, dir_v, float(b))
            np.fill_diagonal(Dc, 0.0)

            flow = fw_beckmann_flow_warm(
                csr,
                edge_cost,
                Dc,
                flow0=row_flow0,
                max_iter=int(FW_MAX_ITER),
                rgap_target=float(FW_RGAP),
                use_numba=bool(USE_NUMBA),
            )

            row_flow0 = flow
            last_flow = flow

            val = flow - flow_base if DELTA else flow
            Z[ia, ib] = flow_norm(val, norm_kind)

            done += 1
            if done == 1 or done % max(1, total // 20) == 0 or done == total:
                print(f"progress: {done}/{total} ({done/total:.0%})")

    if SAVE_NPZ is not None:
        SAVE_NPZ.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            SAVE_NPZ,
            a=a_vals,
            b=b_vals,
            z=Z,
            meta=np.array([meta], dtype=object),
            projection=np.array([projection], dtype=object),
            norm=np.array([norm_kind], dtype=object),
            delta=np.array([bool(DELTA)], dtype=bool),
            n=np.array([n], dtype=int),
            m=np.array([csr.m], dtype=int),
            fw_max_iter=np.array([int(FW_MAX_ITER)], dtype=int),
            fw_rgap=np.array([float(FW_RGAP)], dtype=float),
        )
        print(f"saved grid:   {SAVE_NPZ}")

    A, B = np.meshgrid(b_vals, a_vals)  # note: rows are a, cols are b

    fig = plt.figure(figsize=(12, 5))

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax3d.plot_surface(B, A, Z, cmap="viridis", linewidth=0, antialiased=True)
    ax3d.set_xlabel("a")
    ax3d.set_ylabel("b")
    ax3d.set_zlabel(f"||flow|| ({norm_kind})" + (" (delta)" if DELTA else ""))

    ax2 = fig.add_subplot(1, 2, 2)
    levels = 30
    cs = ax2.contourf(B, A, Z, levels=levels, cmap="viridis")
    fig.colorbar(cs, ax=ax2)
    ax2.set_xlabel("a")
    ax2.set_ylabel("b")

    title = TITLE.strip()
    if not title:
        title = f"Beckmann flow norm surface ({projection}, n={n}, grid={GRID}x{GRID})"
    fig.suptitle(title + "\n" + meta, fontsize=10)

    fig.tight_layout()
    if OUT_PNG is not None:
        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PNG, dpi=160)
        print(f"saved figure: {OUT_PNG}")

    plt.show()


if __name__ == "__main__":
    main()
