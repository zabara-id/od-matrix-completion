from pathlib import Path
from typing import Dict

import numpy as np

from subfunctions import *
from mirror_descent_completion import mirror_descent_completion
from optimize_results_dto import MirrorDescentResult
from src.od_matrix_completion.core.models.manyalli_written_beckmann import fw_beckmann_flow


OD_PATH = Path("data/processed/Mat_Car_ev.csv")


def run_completion(
    D_reference_true: np.ndarray,
    modes_tuple: tuple,
    observed_fraction: float = 0.15,
    reference_noise_level: float = 0.005,
) -> Dict[str, MirrorDescentResult]:
    
    # Референс для KL регуляризатора - слегка зашумленная истинная матрица корреспонденция
    rng = np.random.default_rng(123)
    ref_noise = reference_noise_level * rng.standard_normal(D_reference_true.shape)
    D_reference = D_reference_true * (1.0 + ref_noise)
    D_reference = np.maximum(D_reference, 0.0)
    np.fill_diagonal(D_reference, 0.0)

    # Маршрутный граф
    csr, edge_cost = build_dense_graph(D_reference.shape[0])

    # Параметры для Франк-Вульфа решения модели Бекмана
    fw_hard_kwargs = {"max_iter": 30, "rgap_target": 1e-3, "verbose": False, "use_numba": True}

    # Параметры для Франк-Вульфа решения модели soft-Бекмана
    fw_soft_kwargs = {"max_iter": 30, "theta": 10.0, "delta_rel": 0.02, "delta_abs": 1e-3, "verbose": False, "use_numba": True}

    # Потоки на рёбрах по модели бекмана 
    flow_ref = fw_beckmann_flow(csr, edge_cost, D_reference_true, **fw_hard_kwargs)

    # Маска для удаления информации о части потоков
    rng_mask = np.random.default_rng()
    mask = (rng_mask.random(flow_ref.shape) < observed_fraction).astype(np.float64)
    if mask.sum() == 0:
        mask[0] = 1.0
    f_hat = flow_ref.copy()

    n_iters = 50
    results: Dict[str, MirrorDescentResult] = {}
    
    for mode in modes_tuple:
        print(f"\nЗапуск зеркального спуска в режиме '{mode}' (наблюдается {mask.mean():.0%} потоков)")
        progress_cb = make_progress_printer(mode, total=n_iters)
        switch_cb = make_switch_logger(mode) if mode == "auto_soft" else None
        res = mirror_descent_completion(
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

        results[mode] = res

        final_obj = res.objective_history[-1]
        final_rel = res.rel_l1_history[-1]
        switch_msg = f", switch_iter={res.switch_iter}" if res.switch_iter is not None else ""
        print(f"  final objective={final_obj:.6e}, rel_l1={final_rel:.3e}{switch_msg}")

    return results


def main():
    observed_fraction = 0.20                        # доля известных потоков
    D_reference_true = load_od_matrix(OD_PATH)      # известная матрица корреспонденций
    modes = ("hard", "soft_grad", "auto_soft")      # режимы работы зеркального спуска

    # Запуск моделирования
    results = run_completion(
        D_reference_true,
        observed_fraction=observed_fraction,
        reference_noise_level=0.005,
        mask_seed=42,
        exact_mask=True,
    )

    plot_history(
        {name: res.objective_history for name, res in results.items()},
        Path("plots/objective_curves.png"),
        f"Hard objective (observed {observed_fraction:.0%} flows)",
        "objective",
        semilogy=True,
    )
    plot_history(
        {name: res.rel_l1_history for name, res in results.items()},
        Path("plots/rel_l1_curves.png"),
        f"||D_k - D_ref||_1 / ||D_ref||_1 (observed {observed_fraction:.0%})",
        "relative L1 error",
        semilogy=True,
    )


if __name__ == "__main__":
    main()
