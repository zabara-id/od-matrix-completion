import numpy as np
import heapq
from dataclasses import dataclass
from typing import Tuple

@dataclass
class CSRGraph:
    """
    CSR representation of a directed graph.

    Nodes: 0 .. n-1
    Edges: 0 .. m-1
    """
    n: int                      # number of nodes
    m: int                      # number of edges
    tail: np.ndarray            # (m,) int32, edge start
    head: np.ndarray            # (m,) int32, edge end
    first_out: np.ndarray       # (n+1,) int32, CSR pointer
    out_eid: np.ndarray         # (m,) int32, edge ids sorted by tail

    @staticmethod
    def from_edges(n_nodes: int, tail: np.ndarray, head: np.ndarray) -> "CSRGraph":
        """
        Build CSRGraph from arrays tail/head.
        """
        tail = np.asarray(tail, dtype=np.int32)
        head = np.asarray(head, dtype=np.int32)
        m = int(tail.size)

        # sort edges by tail (stable)
        order = np.argsort(tail, kind="stable").astype(np.int32)
        tail_sorted = tail[order]

        # build CSR indptr
        first_out = np.zeros(n_nodes + 1, dtype=np.int32)
        np.add.at(first_out, tail_sorted + 1, 1)
        first_out = np.cumsum(first_out, dtype=np.int32)

        return CSRGraph(
            n=int(n_nodes),
            m=m,
            tail=tail,
            head=head,
            first_out=first_out,
            out_eid=order,
        )

    def dijkstra(self, weight: np.ndarray, source: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dijkstra shortest paths from source.

        weight : (m,) edge weights
        source : start node

        return:
          dist[v] — shortest distance to v
          pe[v]   — parent edge id to reach v (or -1)
        """
        weight = np.asarray(weight, dtype=np.float64)

        dist = np.full(self.n, np.inf, dtype=np.float64)
        pe = np.full(self.n, -1, dtype=np.int32)

        s = int(source)
        dist[s] = 0.0
        pq = [(0.0, s)]

        while pq:
            du, u = heapq.heappop(pq)
            if du != dist[u]:
                continue
            
            # цикл по исходящим из вершины рёбрам
            for i in range(self.first_out[u], self.first_out[u + 1]):
                e = int(self.out_eid[i])
                v = int(self.head[e])
                nd = du + weight[e]
                # Если значение изменилось, то вершина не помечена и возможно изменился минимум =>
                # она возможный кандидат на минимум
                if nd < dist[v]:
                    dist[v] = nd
                    pe[v] = e
                    heapq.heappush(pq, (nd, v))

        return dist, pe


# ============================================================
# Beckmann model components
# ============================================================

class BRP:
    """
    BPR стоимость проезда по ребру
    """
    def __init__(self, cap, t0, alpha, beta):
        self.cap = np.maximum(cap, 1e-12)
        self.t0 = t0
        self.alpha = alpha
        self.beta = beta

    def __call__(self, flow):
        # Расчет самой функции стоимости на рёбрах
        x = flow / self.cap
        return self.t0 * (1.0 + self.alpha * x**self.beta)
    
    def grad(self, flow):
        # Расчет производной (фактически это матрица Якоби, но 
        # по факту она диагональная, так что возвращается просто вектор)
        # Матрицу получаем через np.diag(result)
        x = flow / self.cap
        return (self.t0 * self.alpha * self.beta / self.cap) * x**(self.beta - 1) 


def stop_criterion(flow, time, sp_cost):
    """
    Остановка по относительной ошибке
    """
    cur_cost = np.dot(flow, time)
    return max(abs((cur_cost - sp_cost) / cur_cost), 0.0)


# ============================================================
# All-or-Nothing assignment (FW direction)
# ============================================================
def aon_assign(csr: CSRGraph, weight: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    All-or-Nothing loading when ALL nodes are zones.

    csr    : CSRGraph with n nodes
    weight : (m,) edge weights
    D      : (n,n) OD matrix over nodes 0..n-1

    return:
      y       : (m,) AON flows
      sp_cost : sum_{o,d} D[o,d] * dist_o[d]
    """
    weight = np.asarray(weight, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    gradient = np.zeros(shape=(csr.m, csr.n * csr.n))

    if D.shape != (csr.n, csr.n):
        raise ValueError(f"D must have shape ({csr.n},{csr.n}), got {D.shape}")

    solution = np.zeros(csr.m, dtype=np.float64)
    total_assignment_cost = 0.0

    for origin in range(csr.n):
        # 1) Считаем кратчайшие пути 
        # для всех остальных вершин
        dist, previous_edges = csr.dijkstra(weight, origin)
        
        # Понимаем, а какая корреспонденция 
        # должна по ним ехать
        corresp_to_assign = D[origin, :]
        # Расчет итоговой стоимости системы
        total_assignment_cost += float(np.dot(corresp_to_assign, dist))

        # 2) Пересчет потоков по путям в потоки по рёбрам
        # 
        # Этот пересчет можно сделать при помощи матричного умножения:
        # solution = PATH_TO_EDGE_INSIDENCE @ assign(corresp_to_assign),
        # где assign(corresp_to_assing) = вектор, в котором корреспонденция поставлена на нужные пути 
        # но для этого нужно хранить матрицу + модифицировать выход алгоритма Дейкстры
        for destanation, correspodence in enumerate(corresp_to_assign):
            cur = int(destanation)
            # раскручивание кратчайшего пути для 
            # добавления соотвествующей корреспонденции на ребро
            while cur != origin:
                e = int(previous_edges[cur])
                if e < 0:
                    break
                solution[e] += correspodence
                # Предполагаемый счет градиента
                gradient[e, origin * csr.n + destanation] = 1
                cur = int(csr.tail[e])

    return solution, total_assignment_cost, gradient


# ============================================================
# Frank–Wolfe (Beckmann UE)
# ============================================================
def fw_beckmann(
    csr: CSRGraph,
    edge_cost,
    D : np.ndarray,
    max_iter=200,
    rgap_target=1e-4,
    verbose=True,
):
    """
    Метод Франка-Вульфа для решения задачи TA по модели Бэкманна

    csr: CSRGraph -- представление графа в формате CSR для нахождения исходящих рёбер,
    edge_cost: callable -- функция стоимости ребра в зависимости от нагрузки,
    D: np.ndarray (n, n), n -- количество вершин в графе (они все и origin, и destination),

    """
    flow = np.zeros(csr.m, dtype=np.float64)
    gradient = np.zeros(shape=(csr.m, csr.n * csr.n))

    for k in range(1, max_iter + 1):
        # Решаем ЛП на заданном множестве
        edge_cost_field = edge_cost(flow)
        y, total_cost_k, gradient_k = aon_assign(csr, edge_cost_field, D)

        # Шаг аглоритма Франка-Вульфа
        gamma = 2.0 / (k + 2.0)
        flow = (1.0 - gamma) * flow + gamma * y
        gradient = (1.0 - gamma) * gradient + gamma * gradient_k

        new_edge_cost_field = edge_cost(flow)
        rg = stop_criterion(flow, new_edge_cost_field, total_cost_k)

        # Это просто забавные лишние штуки
        if verbose and (k == 1 or k % 10 == 0 or rg <= rgap_target):
            print(f"iter={k:4d}  gamma={gamma:.6f}  rgap={rg:.3e} grad_norm={np.linalg.norm(gradient):.6f}")

        if rg <= rgap_target:
            break

    return flow, gradient

# ============================================================
# Example
# 
# 1) Матрица корреспонденции должна быть с нулевой дагональю
# 2) Граф транпортной сети должен быть связный, иначе можно решить две разные задачи
# 3) Функция стоимости ребра должна зависеть только от загрузки 
# этого ребра, иначе задачу оптимизации будет поставить невозможно 
# 
# ============================================================
if __name__ == "__main__":
    # Количество origin и destination (пока что одинаковое кол-во == кол-ву вершин в графе)
    n_nodes = 4

    # Направленные рёбра (u -> v)
    tail = np.array([0, 1, 0, 2, 1, 2, 1, 3, 2, 3], dtype=np.int32)
    head = np.array([1, 0, 2, 0, 2, 1, 3, 1, 3, 2], dtype=np.int32)
    # Собираем граф
    csr = CSRGraph.from_edges(n_nodes, tail, head)

    # Функция стоимости ребра
    cap = np.array([2000, 2000, 1500, 1500, 1500, 1500, 2000, 2000, 1500, 1500], dtype=np.float64)
    t0  = np.array([6, 4, 5, 5, 4, 4, 6, 4, 5, 5], dtype=np.float64)
    alpha = np.full_like(t0, 0.15)
    beta  = np.full_like(t0, 4.0)
    edge_cost = BRP(cap, t0, alpha, beta)

    # OD matrix over ALL nodes (n x n)
    D = np.array([
        [0, 100, 50,  0],
        [80,  0,  20, 10],
        [40, 10,  0, 30],
        [0,  30, 10,  0],
    ], dtype=np.float64)

    flow, time = fw_beckmann(csr, edge_cost, D)

    print("\nFinal link flows / times:")
    for e in range(csr.m):
        print(f"{tail[e]}->{head[e]}  flow={flow[e]:.3f}  time={time[e]:.3f}")

