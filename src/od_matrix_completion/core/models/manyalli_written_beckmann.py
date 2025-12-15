import numpy as np
import heapq
from dataclasses import dataclass
from typing import Tuple

try:
    import numba
    _NUMBA_AVAILABLE = True
    print("NUMBA USE")
except ImportError:  # pragma: no cover - optional dependency
    numba = None
    _NUMBA_AVAILABLE = False

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
# Optional numba-accelerated helpers
# ============================================================
if _NUMBA_AVAILABLE:
    @numba.njit(cache=True)
    def _heap_sift_up(nodes, costs, idx):
        while idx > 0:
            parent = (idx - 1) // 2
            if costs[idx] < costs[parent]:
                tmp_node = nodes[parent]
                tmp_cost = costs[parent]
                nodes[parent] = nodes[idx]
                costs[parent] = costs[idx]
                nodes[idx] = tmp_node
                costs[idx] = tmp_cost
                idx = parent
            else:
                break


    @numba.njit(cache=True)
    def _heap_sift_down(nodes, costs, idx, size):
        while True:
            left = 2 * idx + 1
            right = left + 1
            smallest = idx

            if left < size and costs[left] < costs[smallest]:
                smallest = left
            if right < size and costs[right] < costs[smallest]:
                smallest = right

            if smallest == idx:
                break

            tmp_node = nodes[idx]
            tmp_cost = costs[idx]
            nodes[idx] = nodes[smallest]
            costs[idx] = costs[smallest]
            nodes[smallest] = tmp_node
            costs[smallest] = tmp_cost
            idx = smallest


    @numba.njit(cache=True)
    def _heap_push(nodes, costs, size, node, cost):
        nodes[size] = node
        costs[size] = cost
        _heap_sift_up(nodes, costs, size)
        return size + 1


    @numba.njit(cache=True)
    def _heap_pop(nodes, costs, size):
        node = nodes[0]
        cost = costs[0]
        size -= 1
        if size > 0:
            nodes[0] = nodes[size]
            costs[0] = costs[size]
            _heap_sift_down(nodes, costs, 0, size)
        return node, cost, size


    @numba.njit(cache=True)
    def _dijkstra_numba(n, first_out, out_eid, head, weight, source, dist, pe, heap_nodes, heap_costs):
        dist.fill(np.inf)
        pe.fill(-1)
        heap_size = 0

        heap_size = _heap_push(heap_nodes, heap_costs, heap_size, source, 0.0)
        dist[source] = 0.0

        while heap_size > 0:
            u, du, heap_size = _heap_pop(heap_nodes, heap_costs, heap_size)
            if du != dist[u]:
                continue

            for i in range(first_out[u], first_out[u + 1]):
                e = out_eid[i]
                v = head[e]
                nd = du + weight[e]
                if nd < dist[v]:
                    dist[v] = nd
                    pe[v] = e
                    heap_size = _heap_push(heap_nodes, heap_costs, heap_size, v, nd)

        return dist, pe


    @numba.njit(cache=True)
    def _aon_assign_numba_core(n, m, first_out, out_eid, tail, head, weight, D):
        solution = np.zeros(m, dtype=np.float64)
        gradient = np.zeros((m, n * n), dtype=np.float64)
        total_assignment_cost = 0.0

        dist = np.empty(n, dtype=np.float64)
        pe = np.empty(n, dtype=np.int64)
        heap_nodes = np.empty(m, dtype=np.int64)
        heap_costs = np.empty(m, dtype=np.float64)

        for origin in range(n):
            dist, pe = _dijkstra_numba(n, first_out, out_eid, head, weight, origin, dist, pe, heap_nodes, heap_costs)

            corresp_to_assign = D[origin, :]
            total_assignment_cost += float(np.dot(corresp_to_assign, dist))

            for destination in range(n):
                cur = destination
                while cur != origin:
                    e = pe[cur]
                    if e < 0:
                        break
                    solution[e] += corresp_to_assign[destination]
                    gradient[e, origin * n + destination] = 1.0
                    cur = tail[e]

        return solution, total_assignment_cost, gradient


# ============================================================
# All-or-Nothing assignment (FW direction)
# ============================================================
def aon_assign(csr: CSRGraph, weight: np.ndarray, D: np.ndarray, use_numba: bool = True) -> Tuple[np.ndarray, float]:
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

    if D.shape != (csr.n, csr.n):
        raise ValueError(f"D must have shape ({csr.n},{csr.n}), got {D.shape}")

    if use_numba and _NUMBA_AVAILABLE:
        return _aon_assign_numba(csr, weight, D)

    return _aon_assign_py(csr, weight, D)


def _aon_assign_py(csr: CSRGraph, weight: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Pure NumPy implementation (fallback when numba is unavailable).
    """
    weight = np.asarray(weight, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    gradient = np.zeros(shape=(csr.m, csr.n * csr.n))

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


def _aon_assign_numba(csr: CSRGraph, weight: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    numba-accelerated All-or-Nothing loading.

    Uses a simple binary heap implemented in nopython mode. Falls back to the
    pure NumPy version if numba is not installed.
    """
    # Keep arrays contiguous and avoid implicit upcasting inside numba
    tail = np.asarray(csr.tail, dtype=np.int64)
    head = np.asarray(csr.head, dtype=np.int64)
    first_out = np.asarray(csr.first_out, dtype=np.int64)
    out_eid = np.asarray(csr.out_eid, dtype=np.int64)

    return _aon_assign_numba_core(
        int(csr.n),
        int(csr.m),
        first_out,
        out_eid,
        tail,
        head,
        np.asarray(weight, dtype=np.float64),
        np.asarray(D, dtype=np.float64),
    )


# ============================================================
# Frank–Wolfe (Beckmann UE)
# ============================================================
def fw_beckmann(
    csr: CSRGraph,
    edge_cost,
    D : np.ndarray,
    max_iter=500,
    rgap_target=1e-4,
    verbose=True,
    use_numba=True,
):
    """
    Метод Франка-Вульфа для решения задачи TA по модели Бэкманна

    csr: CSRGraph -- представление графа в формате CSR для нахождения исходящих рёбер,
    edge_cost: callable -- функция стоимости ребра в зависимости от нагрузки,
    D: np.ndarray (n, n), n -- количество вершин в графе (они все и origin, и destination),
    use_numba: bool -- использовать ускоренный AON через numba (если установлена),

    """
    flow = np.zeros(csr.m, dtype=np.float64)
    gradient = np.zeros(shape=(csr.m, csr.n * csr.n))

    for k in range(1, max_iter + 1):
        # Решаем ЛП на заданном множестве
        edge_cost_field = edge_cost(flow)
        y, total_cost_k, gradient_k = aon_assign(csr, edge_cost_field, D, use_numba=use_numba)

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
