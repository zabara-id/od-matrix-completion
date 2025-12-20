import numpy as np
from typing import Callable, Dict, Optional, Tuple

from .manyalli_written_beckmann import CSRGraph

try:  # pragma: no cover - optional dependency
    from .manyalli_written_beckmann import _NUMBA_AVAILABLE, _dijkstra_numba  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _NUMBA_AVAILABLE = False
    _dijkstra_numba = None


def build_incoming_index(csr: CSRGraph) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает (first_in, in_eid) так, что входящие рёбра в v:
      edges = in_eid[first_in[v] : first_in[v+1]]
    """
    order = np.argsort(csr.head, kind="stable").astype(np.int32)
    head_sorted = csr.head[order]
    first_in = np.zeros(csr.n + 1, dtype=np.int32)
    np.add.at(first_in, head_sorted + 1, 1)
    first_in = np.cumsum(first_in, dtype=np.int32)
    return first_in, order


def _prepare_numba_workspace(csr: CSRGraph) -> Optional[Dict[str, np.ndarray]]:
    if not (_NUMBA_AVAILABLE and _dijkstra_numba is not None):
        return None

    return {
        "first_out": np.asarray(csr.first_out, dtype=np.int64),
        "out_eid": np.asarray(csr.out_eid, dtype=np.int64),
        "head": np.asarray(csr.head, dtype=np.int64),
        "dist": np.empty(csr.n, dtype=np.float64),
        "pe": np.empty(csr.n, dtype=np.int64),
        "heap_nodes": np.empty(csr.m, dtype=np.int64),
        "heap_costs": np.empty(csr.m, dtype=np.float64),
    }


def _dijkstra_shortest_paths(
    csr: CSRGraph,
    weight: np.ndarray,
    origin: int,
    *,
    use_numba: bool,
    workspace: Optional[Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вызывает либо numba-ускоренный, либо обычный Дейкстру.
    """
    if use_numba and workspace is not None:
        dist = workspace["dist"]
        pe = workspace["pe"]
        dist, pe = _dijkstra_numba(  # type: ignore[misc]
            int(csr.n),
            workspace["first_out"],
            workspace["out_eid"],
            workspace["head"],
            np.asarray(weight, dtype=np.float64),
            int(origin),
            dist,
            pe,
            workspace["heap_nodes"],
            workspace["heap_costs"],
        )
        return dist, pe

    return csr.dijkstra(weight, origin)


def _soft_aon_prepare_origin(
    csr: CSRGraph,
    first_in: np.ndarray,
    in_eid: np.ndarray,
    weight: np.ndarray,
    origin: int,
    *,
    theta: float,
    delta_rel: float,
    delta_abs: float,
    use_numba: bool,
    workspace: Optional[Dict[str, np.ndarray]],
) -> dict:
    """
    Подготовка кэша для одного origin:
    - dist, pe из Дейкстры
    - order_asc, order_desc по dist
    - для каждого v: cand_edges, cand_tails, cand_probs (по incoming edges)
    """
    dist, pe = _dijkstra_shortest_paths(
        csr, weight, origin, use_numba=use_numba, workspace=workspace
    )
    finite = np.isfinite(dist)
    order_asc = np.argsort(dist, kind="stable")
    order_asc = order_asc[finite[order_asc]]
    order_desc = order_asc[::-1]

    cand_edges = [None] * csr.n
    cand_tails = [None] * csr.n
    cand_probs = [None] * csr.n

    o = int(origin)
    tiny = 1e-12

    for v in order_asc:
        v = int(v)
        if v == o:
            continue

        dv = dist[v]
        if not np.isfinite(dv):
            continue

        delta = delta_abs + delta_rel * float(dv)

        edges = in_eid[first_in[v] : first_in[v + 1]]
        e_list = []
        u_list = []
        w_list = []

        for e in edges:
            e = int(e)
            u = int(csr.tail[e])
            du = dist[u]
            if not np.isfinite(du):
                continue
            # гарантируем убывание dist, чтобы не было циклов
            if du >= dv - tiny:
                continue

            slack = float(du + weight[e] - dv)
            if slack < 0.0:
                slack = 0.0
            if slack <= delta:
                e_list.append(e)
                u_list.append(u)
                w_list.append(np.exp(-theta * slack))

        if not e_list:
            # fallback: строгое дерево кратчайших путей
            e = int(pe[v])
            if e >= 0:
                cand_edges[v] = np.array([e], dtype=np.int32)
                cand_tails[v] = np.array([int(csr.tail[e])], dtype=np.int32)
                cand_probs[v] = np.array([1.0], dtype=np.float64)
            else:
                cand_edges[v] = np.array([], dtype=np.int32)
                cand_tails[v] = np.array([], dtype=np.int32)
                cand_probs[v] = np.array([], dtype=np.float64)
        else:
            ww = np.asarray(w_list, dtype=np.float64)
            s = float(ww.sum())
            if s <= 0.0:
                p = np.full(len(ww), 1.0 / len(ww), dtype=np.float64)
            else:
                p = ww / s
            cand_edges[v] = np.asarray(e_list, dtype=np.int32)
            cand_tails[v] = np.asarray(u_list, dtype=np.int32)
            cand_probs[v] = p

    return {
        "origin": o,
        "dist": np.asarray(dist, dtype=np.float64).copy(),
        "order_asc": order_asc.astype(np.int32),
        "order_desc": order_desc.astype(np.int32),
        "cand_edges": cand_edges,
        "cand_tails": cand_tails,
        "cand_probs": cand_probs,
    }


def soft_aon_assign_logit(
    csr: CSRGraph,
    first_in: np.ndarray,
    in_eid: np.ndarray,
    weight: np.ndarray,
    D: np.ndarray,
    *,
    theta: float = 10.0,
    delta_rel: float = 0.02,
    delta_abs: float = 1e-3,
    use_numba: bool = True,
) -> Tuple[np.ndarray, float, list]:
    """
    Soft AON loading:
    y = expected edge flows under logit choice over near-shortest incoming edges.

    Returns:
      y         : (m,)
      total_cost: sum_e y[e] * weight[e]
      cache     : list of per-origin caches (для VJP/JT)
    """
    weight = np.asarray(weight, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    y = np.zeros(csr.m, dtype=np.float64)
    total_cost = 0.0

    caches = []
    workspace = _prepare_numba_workspace(csr) if use_numba else None

    for origin in range(csr.n):
        cache = _soft_aon_prepare_origin(
            csr,
            first_in,
            in_eid,
            weight,
            origin,
            theta=theta,
            delta_rel=delta_rel,
            delta_abs=delta_abs,
            use_numba=use_numba,
            workspace=workspace,
        )
        caches.append(cache)

        mass = D[origin, :].astype(np.float64, copy=True)
        o = cache["origin"]

        for v in cache["order_desc"]:
            v = int(v)
            if v == o:
                continue
            mv = float(mass[v])
            if mv == 0.0:
                continue

            edges = cache["cand_edges"][v]
            tails = cache["cand_tails"][v]
            probs = cache["cand_probs"][v]

            if edges.size == 0:
                continue

            # распределяем массу на предков
            for e, u, p in zip(edges, tails, probs):
                f = mv * float(p)
                y[int(e)] += f
                mass[int(u)] += f
                total_cost += f * float(weight[int(e)])

    return y, total_cost, caches


def soft_aon_JT_logit(
    csr: CSRGraph,
    caches: list,
    vec_m: np.ndarray,
) -> np.ndarray:
    """
    Возвращает (J_softAON)^T vec_m как матрицу grad_D формы (n,n),
    где J_softAON: D -> y.

    Важно: это JT только для этапа загрузки (AON), при фиксированных weight/кэшах.
    """
    vec_m = np.asarray(vec_m, dtype=np.float64)
    n = int(csr.n)
    grad_D = np.zeros((n, n), dtype=np.float64)

    for cache in caches:
        o = int(cache["origin"])
        order_asc = cache["order_asc"]

        # direct[v] = sum_{e in cand(v)} vec[e] * p(e|v)
        direct = np.zeros(n, dtype=np.float64)
        for v in order_asc:
            v = int(v)
            if v == o:
                continue
            edges = cache["cand_edges"][v]
            probs = cache["cand_probs"][v]
            if edges is None or edges.size == 0:
                continue
            direct[v] = float(np.dot(vec_m[edges], probs))

        # DP: a[v] = direct[v] + sum_{e=u->v} p(e|v) * a[u]
        a = np.zeros(n, dtype=np.float64)
        for v in order_asc:
            v = int(v)
            if v == o:
                continue
            edges = cache["cand_edges"][v]
            tails = cache["cand_tails"][v]
            probs = cache["cand_probs"][v]
            if edges is None or edges.size == 0:
                continue
            acc = direct[v]
            for u, p in zip(tails, probs):
                acc += float(p) * a[int(u)]
            a[v] = acc

        grad_D[o, :] = a

    # диагональ OD не используем
    np.fill_diagonal(grad_D, 0.0)
    return grad_D


def fw_beckmann_soft(
    csr: CSRGraph,
    edge_cost,
    D: np.ndarray,
    *,
    max_iter: int = 30,
    rgap_target: float = 1e-3,
    theta: float = 10.0,
    delta_rel: float = 0.02,
    delta_abs: float = 1e-3,
    verbose: bool = False,
    use_numba: bool = True,
) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """
    Возвращает:
      flow : (m,) финальные потоки
      JT   : функция, которая по vec_m (m,) возвращает приближённый grad_D (n,n) = J^T vec_m

    JT здесь "straight-through":
      - учитывает только линейную зависимость soft assignment от D при фиксированных весах на каждой итерации
      - не дифференцирует через обновление весов, и не дифференцирует dist в Dijkstra
    Это ровно то, что обычно ломается на жёстком AON, а soft AON делает заметно стабильнее.
    """
    D = np.asarray(D, dtype=np.float64)

    first_in, in_eid = build_incoming_index(csr)

    flow = np.zeros(csr.m, dtype=np.float64)
    weights_list = []
    gammas_list = []
    caches_list = []

    for k in range(1, max_iter + 1):
        w = np.asarray(edge_cost(flow), dtype=np.float64)
        y, _, caches = soft_aon_assign_logit(
            csr,
            first_in,
            in_eid,
            w,
            D,
            theta=theta,
            delta_rel=delta_rel,
            delta_abs=delta_abs,
            use_numba=use_numba,
        )

        gamma = 2.0 / (k + 2.0)
        flow = (1.0 - gamma) * flow + gamma * y

        weights_list.append(w)
        gammas_list.append(gamma)
        caches_list.append(caches)

        if verbose and (k == 1 or k % 10 == 0):
            print(f"iter={k:4d} gamma={gamma:.6f}")

        if rgap_target is not None and rgap_target > 0.0:
            # Используем относительный разрыв как в стандартном FW для диагностики
            sp_cost = float(np.dot(y, w))
            cur_cost = float(np.dot(flow, w))
            rg = max(abs((cur_cost - sp_cost) / max(cur_cost, 1e-12)), 0.0)
            if verbose and (k == 1 or k % 10 == 0):
                print(f"      rgap~={rg:.3e}")
            if rg <= rgap_target:
                break

    def JT(vec_m: np.ndarray) -> np.ndarray:
        g = np.asarray(vec_m, dtype=np.float64)
        grad_D = np.zeros((csr.n, csr.n), dtype=np.float64)

        # reverse through convex combinations (straight-through)
        for caches, gamma in zip(reversed(caches_list), reversed(gammas_list)):
            grad_D += soft_aon_JT_logit(csr, caches, gamma * g)
            g = (1.0 - gamma) * g

        np.fill_diagonal(grad_D, 0.0)
        return grad_D

    return flow, JT
