import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List

from aequilibrae.paths import Graph, TrafficClass, TrafficAssignment
from aequilibrae.matrix import AequilibraeMatrix

class BeckmannSolver:
    def __init__(self, links, zones):
        self.matrix_name_ = "demand"
        self.links_ = links
        self.graph_ = None
        self.zones_ =  zones
        # traffic class and traffic ass
        self.tc_ = None
        self.ta_ = None

        # Подготовка к проведению расчетов
        self.construct_graph()


    def construct_graph(self):
        """
        Готовит Graph из таблицы ребер и списка зон.
        Проставляет дефолты и разворачивает двусторонние direction=0 (если есть).
        """
        # обязательные поля
        for col in ["a_node", "b_node", "capacity", "free_flow_time"]:
            if col not in self.links_.columns:
                raise ValueError(f"В links нет обязательной колонки: {col}")

        # дефолтные значения для данных
        if "link_id" not in self.links_.columns:
            self.links_["link_id"] = np.arange(1, len(self.links_) + 1)
        if "direction" not in self.links_.columns:
            self.links_["direction"] = 1
        if "alpha" not in self.links_.columns:
            self.links_["alpha"] = 0.15
        if "beta" not in self.links_.columns:
            self.links_["beta"] = 4.0

        # direction=0 -> добавить обратную дугу в список рёбер графа
        if (self.links_["direction"] == 0).any():
            bi = self.links_["direction"] == 0
            flip = self.links_.loc[bi, ["b_node","a_node","capacity","free_flow_time","alpha","beta"]].copy()
            flip.columns = ["a_node","b_node","capacity","free_flow_time","alpha","beta"]
            flip["direction"] = 1
            flip["link_id"] = self.links_["link_id"].max() + np.arange(1, len(flip) + 1)
            self.links_ = pd.concat([self.links_.loc[~bi].copy(), self.links_.loc[bi].assign(direction=1), flip], ignore_index=True).reset_index(drop=True)

        self.graph_ = Graph()
        self.graph_.network = self.links_
        self.graph_.prepare_graph(self.zones_)                # задаем индексы для OD пар
        self.graph_.set_blocked_centroid_flows(False)
        self.graph_.set_graph("free_flow_time")         # поле стоимости
        

    def _build_ae_matrix(self, D: np.ndarray) -> AequilibraeMatrix:
        """
        Создает AequilibraeMatrix в памяти и записывает OD D (размер Z×Z) c индексацией zones.
        """
        mat = AequilibraeMatrix()
        mat.create_empty(zones=len(self.zones_), matrix_names=[self.matrix_name_], memory_only=True)
        mat.index[:] = self.zones_[:]                     # индексация зон
        mat.matrix[self.matrix_name_][:, :] = D[:, :]
        mat.computational_view([self.matrix_name_])
        return mat
    
    def setup_assignment(self, mat: np.ndarray,
                     vdf: str = "BPR",
                     algo: str = "bfw",
                     max_iter: int = 200,
                     rgap_target: float = 1e-4) -> TrafficAssignment:
        """
        Создает TrafficClass и TrafficAssignment, настраивает VDF/алгоритм и критерии сходимости.
        """
        self.tc_ = TrafficClass(name="car", graph=self.graph_, matrix=self._build_ae_matrix(mat))
        self.ta_ = TrafficAssignment()
        self.ta_.add_class(self.tc_)
        self.ta_.set_vdf(vdf)
        self.ta_.set_vdf_parameters({"alpha": "alpha", "beta": "beta"})
        self.ta_.set_capacity_field("capacity")
        self.ta_.set_time_field("free_flow_time")
        self.ta_.set_algorithm(algo)
        self.ta_.max_iter = max_iter
        self.ta_.rgap_target = rgap_target

    def solve(self):
        self.ta_.execute()

    def flows(self):
        return self.ta_.results().copy()["demand_ab"].to_numpy(dtype=float) 

    def results(self):    
        res = self.ta_.results().copy()
        # Потоки по направлениям (как векторы numpy)
        flow_ab = res["demand_ab"].to_numpy(dtype=float)   # поток a_node -> b_node
        # Суммарный поток по каждому линку (если линки двусторонние)

        flows_df = self.links_[["link_id", "a_node", "b_node"]].copy()
        flows_df["flow"]  = flow_ab
        return flows_df
    
    def congested_time_results(self):
        raise NotImplemented

    def full_results(self):
        return self.ta_.results()