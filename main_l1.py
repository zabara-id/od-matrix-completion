from __future__ import annotations
import networkx as nx
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from mirror_descent_completion_kl import project_to_marginals_masked
from mirror_descent_completion_l1 import mirror_descent_completion_l1
from optimize_results_l1_dto import MirrorDescentL1Result
from subfunctions import build_dense_graph, make_progress_printer, make_switch_logger, plot_history
from src.od_matrix_completion.core.models.manyalli_written_beckmann import BRP, CSRGraph, fw_beckmann_flow


net_file = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/master/SiouxFalls/SiouxFalls_net.tntp"
demand_file = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/master/SiouxFalls/CSV-data/SiouxFalls_od.csv"


def run_completion_l1(
    D_true: np.ndarray,
    modes_tuple: tuple,
    *,
    csr: CSRGraph | None = None,
    edge_cost: BRP | None = None,
    observed_fraction: float = 0.15,
    reference_noise_level: float = 0.,
    reg_lambda: float = 1e-3,
    md_step0: float = 1e-2,
    theta: float = 10.0,
    preserve_true_marginals: bool = True,
    flow_noise_level: float = 0.0,
    flow_noise_seed: int = 123,
    n_iters: int = 30,
    mask_mode: str = "random_mask"
) -> Dict[str, MirrorDescentL1Result]:
    """
    Запускает эксперимент по восстановлению OD-матрицы по частично наблюдаемым потокам.

    Отличие от `main.py`: регуляризация в зеркальном спуске это
        λ * ||D_k - D_true||_1
    вместо KL.
    """
    D_true = np.asarray(D_true, dtype=np.float64)

    rng = np.random.default_rng(123)
    ref_noise = float(reference_noise_level) * rng.standard_normal(D_true.shape)
    D_reference = D_true * (1.0 + ref_noise)
    D_reference = np.maximum(D_reference, 0.0)
    np.fill_diagonal(D_reference, 0.0)

    if preserve_true_marginals:
        allowed = np.ones_like(D_reference, dtype=np.float64)
        np.fill_diagonal(allowed, 0.0)
        L_true = D_true.sum(axis=1)
        W_true = D_true.sum(axis=0)
        D_reference = project_to_marginals_masked(D_reference, L_true, W_true, allowed, n_iters=80)

    if csr is None or edge_cost is None:
        csr, edge_cost = build_dense_graph(D_reference.shape[0])

    fw_hard_kwargs = {"max_iter": 200, "rgap_target": 1e-3, "verbose": False, "use_numba": True}
    fw_soft_kwargs = {
        "max_iter": 200,
        "theta": float(theta),
        "delta_rel": 0.02,
        "delta_abs": 1e-3,
        "verbose": False,
        "use_numba": True,
    }

    flow_ref = fw_beckmann_flow(csr, edge_cost, D_true, **fw_hard_kwargs)

    if mask_mode == "random_mask":
        rng_mask = np.random.default_rng(42)
        mask = (rng_mask.random(flow_ref.shape) < float(observed_fraction)).astype(np.float64)
        if mask.sum() == 0:
            mask[0] = 1.0
    elif mask_mode == "maximal_mask":
        top_idx = int(observed_fraction*100)
        indexes = np.argsort(flow_ref)
        mask = np.zeros(flow_ref.shape)
        mask[indexes[-top_idx:]] = 1
    elif mask_mode == "minimal_mask":
        top_idx = int(observed_fraction*100)
        indexes = np.argsort(flow_ref)
        mask = np.zeros(flow_ref.shape)
        mask[indexes[:top_idx]] = 1

    f_hat = flow_ref.copy()
    if float(flow_noise_level) > 0.0:
        rng_noise = np.random.default_rng(int(flow_noise_seed))
        noise = float(flow_noise_level) * np.maximum(np.abs(flow_ref), 1.0) * rng_noise.standard_normal(flow_ref.shape)
        f_hat = flow_ref + mask * noise

    results: Dict[str, MirrorDescentL1Result] = {}

    for mode in modes_tuple:
        print(f"\nL1-MD запуск в режиме '{mode}' (наблюдается {mask.mean():.0%} потоков)")
        progress_cb = make_progress_printer(mode, total=int(n_iters))
        switch_cb = make_switch_logger(mode) if mode == "auto_soft" else None
        res = mirror_descent_completion_l1(
            csr,
            edge_cost,
            f_hat,
            D_reference=D_reference,
            D_true=D_true,
            mode=mode,
            mask=mask,
            reg_lambda=float(reg_lambda),
            n_iters=int(n_iters),
            step0=float(md_step0),
            fw_hard_kwargs=fw_hard_kwargs,
            fw_soft_kwargs=fw_soft_kwargs,
            progress_callback=progress_cb,
            switch_callback=switch_cb,
        )

        results[mode] = res

        final_obj = res.objective_history[-1]
        final_data = res.data_history[-1] if res.data_history else float("nan")
        final_reg = res.l1_to_true_history[-1]
        final_rel_ref = res.rel_l1_ref_history[-1]
        final_rel_true = res.rel_l1_true_history[-1]
        accept_rate = float(np.mean(res.accepted_history)) if res.accepted_history else float("nan")
        last_step = res.step_history[-1] if res.step_history else float("nan")
        last_trials = res.ls_trials_history[-1] if res.ls_trials_history else -1
        switch_msg = f", switch_iter={res.switch_iter}" if res.switch_iter is not None else ""
        print(
            f"  final objective={final_obj:.6e} (data={final_data:.6e}, l1={final_reg:.6e}), "
            f"rel_l1_ref={final_rel_ref:.3e}, rel_l1_true={final_rel_true:.3e}, "
            f"accept={accept_rate:.0%}, last_step={last_step:.2e}, last_ls_trials={last_trials}{switch_msg}"
        )

    return results


def main():
    dem = pd.read_csv(demand_file)
    zones = int(max(dem.O.max(), dem.D.max()))

    D_true = np.zeros(shape=(zones, zones), dtype=float)
    for element in dem.to_records(index=False):
        D_true[element[0] - 1][element[1] - 1] = element[2]

    net = pd.read_csv(net_file, skiprows=2, sep="\t", lineterminator=";", header=None)
    net.columns = [
        "newline",
        "a_node",
        "b_node",
        "capacity",
        "length",
        "free_flow_time",
        "b",
        "power",
        "speed",
        "toll",
        "link_type",
        "terminator",
    ]
    net.drop(columns=["newline", "terminator"], index=[76], inplace=True)
    network = net[["a_node", "b_node", "capacity", "free_flow_time"]]

    network = network.assign(direction=1)
    network["link_id"] = network.index
    network = network.astype({"a_node": "int64", "b_node": "int64"})

    graph = CSRGraph.from_edges(
        n_nodes=24,
        tail=network["a_node"].to_numpy() - 1,
        head=network["b_node"].to_numpy() - 1,
    )

    nx.draw(graph.to_networkx())
    plt.gcf().suptitle("Икша")
    # plt.show()

    edge_cost = BRP(
        cap=network["capacity"].to_numpy(),
        t0=network["free_flow_time"].to_numpy(),
        alpha=0.15,
        beta=4,
    )

    observed_fractions = (0.1, )
    modes = ("hard", )

    # mask_modes = ("maximal_mask", )

    np.random.seed(42)
    results_by_label: Dict[str, MirrorDescentL1Result] = {}
    for observed_fraction in observed_fractions:
        results = run_completion_l1(
            D_true,
            modes,
            csr=graph,
            n_iters=40,
            edge_cost=edge_cost,
            observed_fraction=float(observed_fraction),
            reference_noise_level=0.2,
            reg_lambda=5e2,
            mask_mode="maximal_mask"
        )
        for _, res in results.items():
            label = f"obs={observed_fraction:.0%} max flows"
            results_by_label[label] = res
    
    # results_by_label: Dict[str, MirrorDescentL1Result] = {}
    # for mask_mode in mask_modes:
    #     results = run_completion_l1(
    #         D_true,
    #         modes,
    #         csr=graph,
    #         n_iters=450,
    #         mask_mode=mask_mode,
    #         edge_cost=edge_cost,
    #         observed_fraction=float(observed_fraction),
    #         reference_noise_level=0.0,
    #         reg_lambda=5e2,
    #     )
    #     for _, res in results.items():
    #         label = f"{observed_fraction:.0%} {mask_mode}"
    #         results_by_label[label] = res

    out_dir = Path("plots_top_10_best_with_noise")
    out_dir.mkdir(parents=True, exist_ok=True)

    # plot_history(
    #     {name: res.objective_history for name, res in results_by_label.items()},
    #     out_dir / "objective_curves.png",
    #     r"$|| \  f(D) - \hat{f}  \ ||_2^2 + \lambda\cdot|| \ D - D_{ref}  \ ||_1$",
    #     "objective",
    #     semilogy=True,
    # )
    plot_history(
        {name: res.data_history for name, res in results_by_label.items()},
        out_dir / "relative_flow_error.png",
        r"$|| \  f(D) - \hat{f} \  ||_1  \ /  \ || \  \hat{f} \ ||_1$",
        "data",
        semilogy=True,
    )
    plot_history(
        {name: res.rel_l1_true_history for name, res in results_by_label.items()},
        out_dir / "rel_matrix_error.png",
        r"$|| \ D_k - D_{true}\ ||_1 \  / \  || \ D_{true} \ ||_1$",
        "relative L1 error",
        semilogy=True,
    )


if __name__ == "__main__":
    main()
