from pathlib import Path
from typing import Dict

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from subfunctions import *
from mirror_descent_completion import mirror_descent_completion, project_to_marginals_masked
from optimize_results_dto import MirrorDescentResult
from src.od_matrix_completion.core.models.manyalli_written_beckmann import fw_beckmann_flow

from src.od_matrix_completion.core.models.manyalli_written_beckmann import CSRGraph, BRP


OD_PATH = Path("data/processed/Mat_Car_ev.csv")
net_file = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/master/SiouxFalls/SiouxFalls_net.tntp"
demand_file = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/master/SiouxFalls/CSV-data/SiouxFalls_od.csv"

def run_completion(
    D_reference_true: np.ndarray,
    modes_tuple: tuple,
    csr = None,
    edge_cost = None,
    observed_fraction: float = 0.15,
    reference_noise_level: float = 0.005,
    reg_lambda: float = 1e-3,
    md_step0: float = 1e-2,
    theta: float = 10.0,
    mask_seed: int = 123,
    preserve_true_marginals: bool = True,
    flow_noise_level: float = 0.0,
    flow_noise_seed: int = 123,
) -> Dict[str, MirrorDescentResult]:
    """Запускает эксперимент по восстановлению OD-матрицы по частично наблюдаемым потокам.

    Строит маршрутный граф, считает истинные потоки ``f_hat`` из ``D_reference_true``,
    затем скрывает часть компонент потока с помощью маски (наблюдается доля ``observed_fraction``)
    и для каждого режима из ``modes_tuple`` запускает ``mirror_descent_completion``.

    Args:
        D_reference_true (np.ndarray): Истинная OD-матрица (форма ``(n, n)``), по которой генерируются наблюдения.

        modes_tuple (tuple): Набор режимов зеркального спуска (например: ``("hard", "soft_grad", "auto_soft")``).

        observed_fraction (float, optional): Доля наблюдаемых компонент потока (от 0 до 1). По умолчанию 0.15.

        reference_noise_level (float, optional): Уровень шума для построения ``D_reference`` (референс для KL), как гауссов шум. По умолчанию 0.005.

    Returns:
        Dict[str, MirrorDescentResult]: Словарь ``mode -> результат`` (траектория целевой функции и метрик, финальные значения).
    """
    
    # Референс для KL регуляризатора - слегка зашумленная истинная матрица корреспонденция
    rng = np.random.default_rng(123)
    # ref_noise = reference_noise_level * rng.standard_normal(D_reference_true.shape)
    ref_noise = 0
    D_reference = D_reference_true * (1.0 + ref_noise)
    D_reference = np.maximum(D_reference, 0.0)
    np.fill_diagonal(D_reference, 0.0)

    if preserve_true_marginals:
        allowed = np.ones_like(D_reference, dtype=np.float64)
        np.fill_diagonal(allowed, 0.0)
        L_true = D_reference_true.sum(axis=1)
        W_true = D_reference_true.sum(axis=0)
        D_reference = project_to_marginals_masked(D_reference, L_true, W_true, allowed, n_iters=80)

    # Маршрутный граф
    if csr == None or edge_cost == None:
        csr, edge_cost = build_dense_graph(D_reference.shape[0])


    # Параметры для Франк-Вульфа решения модели Бекмана
    fw_hard_kwargs = {
        "max_iter": 200,
        "rgap_target": 1e-3,
        "verbose": False,
        "use_numba": True
    }

    # Параметры для Франк-Вульфа решения модели soft-Бекмана
    fw_soft_kwargs = {
        "max_iter": 200,
        "theta": float(theta),
        "delta_rel": 0.02,
        "delta_abs": 1e-3,
        "verbose": False,
        "use_numba": True,
    }

    # Потоки на рёбрах по модели бекмана 
    flow_ref = fw_beckmann_flow(csr, edge_cost, D_reference_true, **fw_hard_kwargs)

    # Маска для удаления информации о части потоков
    rng_mask = np.random.default_rng(int(mask_seed))
    mask = (rng_mask.random(flow_ref.shape) < observed_fraction).astype(np.float64)
    if mask.sum() == 0:
        mask[0] = 1.0

    f_hat = flow_ref.copy()
    if flow_noise_level > 0.0:
        rng_noise = np.random.default_rng(int(flow_noise_seed))
        noise = float(flow_noise_level) * np.maximum(np.abs(flow_ref), 1.0) * rng_noise.standard_normal(flow_ref.shape)
        f_hat = flow_ref + mask * noise

    n_iters = 30                                    # кол-во итераций зеркального спуска
    results: Dict[str, MirrorDescentResult] = {}    # словарь хранения результатов
    
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
            reg_lambda=float(reg_lambda),
            D_prior=D_reference,
            D_target=D_reference_true,
            n_iters=n_iters,
            step0=float(md_step0),
            fw_hard_kwargs=fw_hard_kwargs,
            fw_soft_kwargs=fw_soft_kwargs,
            progress_callback=progress_cb,
            switch_callback=switch_cb,
        )

        results[mode] = res

        final_obj = res.objective_history[-1]
        final_data = res.data_history[-1] if res.data_history else float("nan")
        final_kl = res.kl_history[-1]
        final_rel = res.rel_l1_history[-1]
        final_rel_true = (
            res.rel_l1_target_history[-1] if res.rel_l1_target_history is not None else float("nan")
        )
        accept_rate = float(np.mean(res.accepted_history)) if res.accepted_history else float("nan")
        last_step = res.step_history[-1] if res.step_history else float("nan")
        last_trials = res.ls_trials_history[-1] if res.ls_trials_history else -1
        switch_msg = f", switch_iter={res.switch_iter}" if res.switch_iter is not None else ""
        print(
            f"  final objective={final_obj:.6e} (data={final_data:.6e}, kl={final_kl:.6e}), "
            f"rel_l1_ref={final_rel:.3e}, rel_l1_true={final_rel_true:.3e}, "
            f"accept={accept_rate:.0%}, last_step={last_step:.2e}, last_ls_trials={last_trials}{switch_msg}"
        )

    return results

import pandas as pd

def main():
    # OD matrix
    dem = pd.read_csv(demand_file)
    zones = int(max(dem.O.max(), dem.D.max()))
    index = np.arange(zones) + 1
    D_reference_true = np.zeros(shape=(zones, zones))
    for element in dem.to_records(index=False):
        D_reference_true[element[0]-1][element[1]-1] = element[2]
    
    # Network itsels
    net = pd.read_csv(net_file, skiprows=2, sep="\t", lineterminator=";", header=None)
    net.columns = ["newline", "a_node", "b_node", "capacity", "length", "free_flow_time", "b", "power", "speed", "toll", "link_type", "terminator"]
    net.drop(columns=["newline", "terminator"], index=[76], inplace=True)
    network = net[['a_node', 'b_node', "capacity", 'free_flow_time']]

    network = network.assign(direction=1)
    network["link_id"] = network.index
    network = network.astype({"a_node":"int64", "b_node": "int64"})
 
    graph = CSRGraph.from_edges(n_nodes=24, tail=network['a_node'].to_numpy() - 1, head=network['b_node'].to_numpy() - 1)

    edge_cost = BRP(cap=network['capacity'].to_numpy(), 
                    t0=network['free_flow_time'].to_numpy(),
                    alpha=0.15, beta=4)
    
    
    observed_fraction = 0.8                           # доля известных потоков

    # modes = ("hard", "soft_grad", "auto_soft")      # режимы работы зеркального спуска
    modes = ("hard", )      # режимы работы зеркального спуска

    # Запуск моделирования
    results = run_completion(
        D_reference_true,
        modes,
        csr=graph, edge_cost=edge_cost,
        observed_fraction=observed_fraction,
        reference_noise_level=0.00,
    )

    np.savetxt(
        "matrix.csv",
        results["hard"].D_final,
        delimiter=","
    )
    
    plot_history(
        {name: res.objective_history for name, res in results.items()},
        Path("plots/objective_curves.png"),
        f"Objective = data + λ·KL (observed {observed_fraction:.0%} flows)",
        "objective",
        semilogy=True,
    )
    plot_history(
        {name: res.data_history for name, res in results.items()},
        Path("plots/data_curves.png"),
        f"Data term on observed flows (observed {observed_fraction:.0%})",
        "data",
        semilogy=True,
    )
    plot_history(
        {name: res.kl_history for name, res in results.items()},
        Path("plots/kl_curves.png"),
        f"KL(D || D_prior) (observed {observed_fraction:.0%})",
        "kl",
        semilogy=True,
    )
    plot_history(
        {name: res.rel_l1_history for name, res in results.items()},
        Path("plots/rel_l1_curves.png"),
        f"||D_k - D_ref||_1 / ||D_ref||_1 (observed {observed_fraction:.0%})",
        "relative L1 error",
        semilogy=True,
    )
    plot_history(
        {
            name: res.rel_l1_target_history
            for name, res in results.items()
            if res.rel_l1_target_history is not None
        },
        Path("plots/rel_l1_true_curves.png"),
        f"||D_k - D_true||_1 / ||D_true||_1 (observed {observed_fraction:.0%})",
        "relative L1 error (to true)",
        semilogy=True,
    )


if __name__ == "__main__":
    main()
