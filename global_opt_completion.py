from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import differential_evolution

from mirror_descent_completion import kl_value_and_grad, project_to_marginals_masked
from subfunctions import build_dense_graph, load_od_matrix, plot_history
from src.od_matrix_completion.core.models.manyalli_written_beckmann import fw_beckmann_flow

# ==========================
# Настройки эксперимента
# ==========================
OD_PATH = Path("data/processed/Mat_Car_ev.csv")
OUT_DIR = Path("plots_global")

OBSERVED_FRACTION = 1
REFERENCE_NOISE_LEVEL = 0.005
PRESERVE_TRUE_MARGINALS = True

REG_LAMBDA = 1e-3
MASK_SEED = 123
FLOW_NOISE_LEVEL = 0.0
FLOW_NOISE_SEED = 123

# Опционально: ускорить эксперимент, взяв только первые N зон (None => все зоны).
N_ZONES: int | None = None
SUBSET: str = "first"  # "first" or "random"
SUBSET_SEED = 123

# ==========================
# Настройки глобальной оптимизации
# ==========================
# Параметризация через набор непересекающихся 2x2 cycle-move (сохраняют маргиналии).
N_CYCLES = 24
REL = 0.25
MIN_CELL_VALUE = 1e-6
MOVE_SEED = 123

# Differential Evolution (SciPy) параметры.
DE_SEED = 123
DE_MAXITER = 200
DE_POPSIZE = 3  # реальный размер популяции = DE_POPSIZE * dim
DE_POLISH = False

# ==========================
# Настройки FW (Beckmann)
# ==========================
FW_TRUE_KWARGS = {"max_iter": 200, "rgap_target": 1e-3, "verbose": False, "use_numba": True}
# Для глобальной оптимизации можно снижать max_iter для ускорения (цена: чуть менее точная цель).
FW_OPT_KWARGS = {"max_iter": 60, "rgap_target": 1e-3, "verbose": False, "use_numba": True}
# Финальная точная оценка на найденном решении.
FW_FINAL_KWARGS = {"max_iter": 200, "rgap_target": 1e-3, "verbose": False, "use_numba": True}

PLOT_EPS = 1e-12


@dataclass(frozen=True)
class CycleMove:
    """
    2x2 cycle move, сохраняющий строки/столбцы:
      +a at (o1,d1) and (o2,d2)
      -a at (o1,d2) and (o2,d1)
    """

    o1: int
    o2: int
    d1: int
    d2: int

    def cells(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        return (self.o1, self.d1), (self.o2, self.d2), (self.o1, self.d2), (self.o2, self.d1)


def _subset_od(D: np.ndarray) -> np.ndarray:
    if N_ZONES is None:
        return D
    n_full = int(D.shape[0])
    n = min(int(N_ZONES), n_full)
    if n < 3:
        raise ValueError("N_ZONES must be >= 3")
    if SUBSET not in {"first", "random"}:
        raise ValueError("SUBSET must be 'first' or 'random'")

    if SUBSET == "first":
        idx = np.arange(n, dtype=int)
    else:
        rng = np.random.default_rng(int(SUBSET_SEED))
        idx = np.sort(rng.choice(n_full, size=n, replace=False))

    D_sub = np.asarray(D[np.ix_(idx, idx)], dtype=np.float64).copy()
    np.fill_diagonal(D_sub, 0.0)
    return D_sub


def _sample_disjoint_cycle_moves(
    D0: np.ndarray,
    rng: np.random.Generator,
    *,
    n_moves: int,
    min_value: float,
    max_tries: int = 200_000,
) -> list[CycleMove]:
    n = int(D0.shape[0])
    used: set[tuple[int, int]] = set()
    moves: list[CycleMove] = []

    for _ in range(int(max_tries)):
        if len(moves) >= int(n_moves):
            break

        o1, o2 = rng.choice(n, size=2, replace=False).tolist()
        d1, d2 = rng.choice(n, size=2, replace=False).tolist()

        # Avoid touching diagonal entries.
        if o1 in (d1, d2) or o2 in (d1, d2):
            continue

        mv = CycleMove(o1=o1, o2=o2, d1=d1, d2=d2)
        cells = mv.cells()
        if any(c in used for c in cells):
            continue

        values = [float(D0[i, j]) for i, j in cells]
        if min(values) <= float(min_value):
            continue

        moves.append(mv)
        used.update(cells)

    if len(moves) < int(n_moves):
        raise RuntimeError(
            f"Failed to sample {n_moves} disjoint cycle moves (got {len(moves)}). "
            "Try уменьшить N_CYCLES, REL или поменять MOVE_SEED."
        )
    return moves


def _moves_to_index_arrays(moves: list[CycleMove]) -> tuple[np.ndarray, np.ndarray]:
    rows = np.empty((len(moves), 4), dtype=np.int64)
    cols = np.empty((len(moves), 4), dtype=np.int64)
    for k, mv in enumerate(moves):
        # order: (+,+,-,-)
        rows[k, 0] = mv.o1
        cols[k, 0] = mv.d1
        rows[k, 1] = mv.o2
        cols[k, 1] = mv.d2
        rows[k, 2] = mv.o1
        cols[k, 2] = mv.d2
        rows[k, 3] = mv.o2
        cols[k, 3] = mv.d1
    return rows, cols


def _bounds_for_moves(D0: np.ndarray, moves: list[CycleMove], *, rel: float) -> list[tuple[float, float]]:
    rows, cols = _moves_to_index_arrays(moves)
    rel = float(rel)
    if not (0.0 < rel <= 1.0):
        raise ValueError("REL must be in (0, 1].")

    # (+,+,-,-) signs: negative x decreases (+) cells, positive x decreases (-) cells.
    pos_min = np.min(D0[rows[:, :2], cols[:, :2]], axis=1)
    neg_min = np.min(D0[rows[:, 2:], cols[:, 2:]], axis=1)

    lower = -rel * pos_min
    upper = rel * neg_min
    bounds = [(float(lo), float(hi)) for lo, hi in zip(lower.tolist(), upper.tolist())]

    for i, (lo, hi) in enumerate(bounds):
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise RuntimeError(f"Bad bounds for move[{i}]: ({lo}, {hi})")
    return bounds


class _DEObjective:
    def __init__(
        self,
        *,
        csr,
        edge_cost,
        f_hat: np.ndarray,
        mask: Optional[np.ndarray],
        D0: np.ndarray,
        moves_rows: np.ndarray,
        moves_cols: np.ndarray,
        reg_lambda: float,
        D_prior: np.ndarray,
        D_reference: np.ndarray,
        D_target: Optional[np.ndarray],
        fw_kwargs: dict,
        kl_eps: float = 1e-12,
    ):
        self.csr = csr
        self.edge_cost = edge_cost
        self.f_hat = np.asarray(f_hat, dtype=np.float64)
        self.mask = None if mask is None else np.asarray(mask, dtype=np.float64)
        self.D0 = np.asarray(D0, dtype=np.float64)
        self.rows = np.asarray(moves_rows, dtype=np.int64)
        self.cols = np.asarray(moves_cols, dtype=np.int64)
        self.sign = np.asarray([+1.0, +1.0, -1.0, -1.0], dtype=np.float64)
        self.reg_lambda = float(reg_lambda)
        self.D_prior = np.asarray(D_prior, dtype=np.float64)
        self.D_reference = np.asarray(D_reference, dtype=np.float64)
        self.D_target = None if D_target is None else np.asarray(D_target, dtype=np.float64)
        self.fw_kwargs = dict(fw_kwargs)
        self.kl_eps = float(kl_eps)

        n = int(self.D0.shape[0])
        self.allowed = np.ones((n, n), dtype=np.float64)
        np.fill_diagonal(self.allowed, 0.0)

        self.ref_l1 = float(np.sum(np.abs(self.D_reference * self.allowed)))
        self.target_l1 = (
            float(np.sum(np.abs(self.D_target * self.allowed))) if self.D_target is not None else None
        )

        if self.reg_lambda > 0.0:
            self.D_prior = np.maximum(self.D_prior, self.kl_eps) * self.allowed
            np.fill_diagonal(self.D_prior, 0.0)

        self.n_evals = 0

        self.best_obj = float("inf")
        self.best_x: Optional[np.ndarray] = None
        self.best_D: Optional[np.ndarray] = None
        self.best_flow: Optional[np.ndarray] = None
        self.best_data = float("inf")
        self.best_kl = float("inf")
        self.best_rel_l1 = float("inf")
        self.best_rel_l1_target: Optional[float] = None

        self.objective_history: list[float] = []
        self.data_history: list[float] = []
        self.kl_history: list[float] = []
        self.rel_l1_history: list[float] = []
        self.rel_l1_target_history: Optional[list[float]] = [] if self.D_target is not None else None
        self.eval_count_history: list[int] = []

    def _build_D(self, x: np.ndarray) -> np.ndarray:
        D = self.D0.copy()
        x = np.asarray(x, dtype=np.float64)
        D[self.rows, self.cols] += x[:, None] * self.sign[None, :]
        return D

    def _rel_l1(self, D: np.ndarray) -> float:
        diff = float(np.sum(np.abs((D - self.D_reference) * self.allowed)))
        return diff / max(self.ref_l1, 1e-12)

    def _rel_l1_target(self, D: np.ndarray) -> float:
        if self.D_target is None:
            raise RuntimeError("D_target is None")
        diff = float(np.sum(np.abs((D - self.D_target) * self.allowed)))
        return diff / max(float(self.target_l1 or 0.0), 1e-12)

    def __call__(self, x: np.ndarray) -> float:
        self.n_evals += 1
        D = self._build_D(x)
        flow = fw_beckmann_flow(self.csr, self.edge_cost, D, **self.fw_kwargs)

        residual = flow - self.f_hat
        if self.mask is not None:
            residual = residual * self.mask
        data = 0.5 * float(np.dot(residual, residual))
        kl_val, _ = kl_value_and_grad(D, self.D_prior, self.allowed, eps=self.kl_eps)
        obj = data + self.reg_lambda * float(kl_val)

        if obj < self.best_obj:
            self.best_obj = float(obj)
            self.best_x = np.asarray(x, dtype=np.float64).copy()
            self.best_D = D
            self.best_flow = flow
            self.best_data = float(data)
            self.best_kl = float(kl_val)
            self.best_rel_l1 = self._rel_l1(D)
            if self.D_target is not None:
                self.best_rel_l1_target = self._rel_l1_target(D)

        return float(obj)

    def snapshot_best(self) -> None:
        self.objective_history.append(float(self.best_obj))
        self.data_history.append(float(self.best_data))
        self.kl_history.append(float(self.best_kl))
        self.rel_l1_history.append(float(self.best_rel_l1))
        if self.rel_l1_target_history is not None:
            self.rel_l1_target_history.append(float(self.best_rel_l1_target or 0.0))
        self.eval_count_history.append(int(self.n_evals))


def solve_de_cycles(
    csr,
    edge_cost,
    f_hat: np.ndarray,
    *,
    D_reference: np.ndarray,
    mask: Optional[np.ndarray],
    reg_lambda: float,
    D_prior: np.ndarray,
    D_target: Optional[np.ndarray],
) -> tuple[_DEObjective, np.ndarray, np.ndarray]:
    D_reference = np.asarray(D_reference, dtype=np.float64)
    n = int(D_reference.shape[0])
    allowed = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(allowed, 0.0)

    D0 = np.maximum(D_reference, 0.0) * allowed
    D_prior = np.maximum(D_prior, 0.0) * allowed

    rng = np.random.default_rng(int(MOVE_SEED))
    moves = _sample_disjoint_cycle_moves(D0, rng, n_moves=int(N_CYCLES), min_value=float(MIN_CELL_VALUE))
    bounds = _bounds_for_moves(D0, moves, rel=float(REL))

    rows, cols = _moves_to_index_arrays(moves)
    objective = _DEObjective(
        csr=csr,
        edge_cost=edge_cost,
        f_hat=f_hat,
        mask=mask,
        D0=D0,
        moves_rows=rows,
        moves_cols=cols,
        reg_lambda=float(reg_lambda),
        D_prior=D_prior,
        D_reference=D_reference,
        D_target=D_target,
        fw_kwargs=FW_OPT_KWARGS,
    )

    x0 = np.zeros(len(bounds), dtype=np.float64)
    objective(x0)
    objective.snapshot_best()

    def _cb(_xk: np.ndarray, _convergence: float) -> bool:
        objective.snapshot_best()
        it = len(objective.objective_history) - 1
        if it == 1 or it % 5 == 0:
            print(
                f"[DE-cycles] gen={it:3d} best_obj={objective.best_obj:.6e} "
                f"(data={objective.best_data:.3e}, kl={objective.best_kl:.3e}) evals={objective.n_evals}"
            )
        return False

    de_result = differential_evolution(
        objective,
        bounds=bounds,
        seed=int(DE_SEED),
        maxiter=int(DE_MAXITER),
        popsize=int(DE_POPSIZE),
        polish=bool(DE_POLISH),
        callback=_cb,
        disp=False,
        updating="deferred",
        workers=1,
    )

    if objective.best_x is None or objective.best_D is None:
        raise RuntimeError("DE finished without a best solution (unexpected).")

    D_best = objective.best_D
    flow_best_final = fw_beckmann_flow(csr, edge_cost, D_best, **FW_FINAL_KWARGS)

    print("\n[DE-cycles] SciPy status:", de_result.message)
    print(
        f"[DE-cycles] best objective={objective.best_obj:.6e} "
        f"(data={objective.best_data:.6e}, kl={objective.best_kl:.6e}), "
        f"rel_l1_ref={objective.best_rel_l1:.3e}"
        + (
            f", rel_l1_true={float(objective.best_rel_l1_target or 0.0):.3e}"
            if objective.best_rel_l1_target is not None
            else ""
        )
    )
    print(f"[DE-cycles] evals={objective.n_evals}, dim={len(bounds)}, n={n}, edges m={csr.m}")

    return objective, D_best, flow_best_final


def main() -> None:
    observed_fraction = float(OBSERVED_FRACTION)
    if not (0.0 < observed_fraction <= 1.0):
        raise ValueError("OBSERVED_FRACTION must be in (0, 1].")

    D_true_full = load_od_matrix(OD_PATH)
    D_true = _subset_od(D_true_full)

    # Референс/приор для KL: слегка зашумлённая истинная OD.
    rng_ref = np.random.default_rng(123)
    ref_noise = float(REFERENCE_NOISE_LEVEL) * rng_ref.standard_normal(D_true.shape)
    D_reference = D_true * (1.0 + ref_noise)
    D_reference = np.maximum(D_reference, 0.0)
    np.fill_diagonal(D_reference, 0.0)

    if bool(PRESERVE_TRUE_MARGINALS):
        allowed = np.ones_like(D_reference, dtype=np.float64)
        np.fill_diagonal(allowed, 0.0)
        L_true = D_true.sum(axis=1)
        W_true = D_true.sum(axis=0)
        D_reference = project_to_marginals_masked(D_reference, L_true, W_true, allowed, n_iters=80)

    csr, edge_cost = build_dense_graph(D_reference.shape[0])

    flow_ref = fw_beckmann_flow(csr, edge_cost, D_true, **FW_TRUE_KWARGS)

    rng_mask = np.random.default_rng(int(MASK_SEED))
    mask = (rng_mask.random(flow_ref.shape) < observed_fraction).astype(np.float64)
    if mask.sum() == 0:
        mask[0] = 1.0

    f_hat = flow_ref.copy()
    if float(FLOW_NOISE_LEVEL) > 0.0:
        rng_noise = np.random.default_rng(int(FLOW_NOISE_SEED))
        noise = float(FLOW_NOISE_LEVEL) * np.maximum(np.abs(flow_ref), 1.0) * rng_noise.standard_normal(flow_ref.shape)
        f_hat = flow_ref + mask * noise

    print("=== Global completion via Differential Evolution (cycle moves) ===")
    print(f"OD: n={D_reference.shape[0]}, edges m={csr.m}")
    print(f"observed_fraction={mask.mean():.0%}, reg_lambda={REG_LAMBDA:g}")
    print(f"N_CYCLES={N_CYCLES}, REL={REL:g}, DE_MAXITER={DE_MAXITER}, DE_POPSIZE={DE_POPSIZE}")
    print(f"FW_OPT_MAX_ITER={FW_OPT_KWARGS['max_iter']}, FW_FINAL_MAX_ITER={FW_FINAL_KWARGS['max_iter']}")

    objective, D_best, flow_best = solve_de_cycles(
        csr,
        edge_cost,
        f_hat,
        D_reference=D_reference,
        mask=mask,
        reg_lambda=float(REG_LAMBDA),
        D_prior=D_reference,
        D_target=D_true,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _safe_log(values: list[float]) -> list[float]:
        y = np.asarray(values, dtype=np.float64)
        y = np.maximum(y, float(PLOT_EPS))
        return y.tolist()

    label = "de_cycles"
    plot_history(
        {label: _safe_log(objective.objective_history)},
        OUT_DIR / "objective_curves.png",
        f"Objective = data + λ·KL (observed {observed_fraction:.0%} flows)",
        "objective",
        semilogy=True,
    )
    plot_history(
        {label: _safe_log(objective.data_history)},
        OUT_DIR / "data_curves.png",
        f"Data term on observed flows (observed {observed_fraction:.0%})",
        "data",
        semilogy=True,
    )
    plot_history(
        {label: _safe_log(objective.kl_history)},
        OUT_DIR / "kl_curves.png",
        f"KL(D || D_prior) (observed {observed_fraction:.0%})",
        "kl",
        semilogy=True,
    )
    plot_history(
        {label: _safe_log(objective.rel_l1_history)},
        OUT_DIR / "rel_l1_curves.png",
        f"||D_k - D_ref||_1 / ||D_ref||_1 (observed {observed_fraction:.0%})",
        "relative L1 error",
        semilogy=True,
    )
    if objective.rel_l1_target_history is not None:
        plot_history(
            {label: _safe_log(objective.rel_l1_target_history)},
            OUT_DIR / "rel_l1_true_curves.png",
            f"||D_k - D_true||_1 / ||D_true||_1 (observed {observed_fraction:.0%})",
            "relative L1 error (to true)",
            semilogy=True,
        )

    # Простой sanity print: нормы разницы по OD и по потокам.
    print("\nSanity checks:")
    print("  ||D_best - D_ref||_F =", float(np.linalg.norm(D_best - D_reference)))
    print("  ||flow_best - flow_hat||_2 (masked) =", float(np.linalg.norm((flow_best - f_hat) * mask)))


if __name__ == "__main__":
    main()
