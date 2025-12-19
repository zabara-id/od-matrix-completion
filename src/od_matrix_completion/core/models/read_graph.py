from pathlib import Path
import csv
import numpy as np
import manyalli_written_beckmann as beckmann
import matplotlib.pyplot as plt
import networkx as nx

def remap_edges_csv(in_csv: str | Path, out_csv: str | Path, encoding: str = "utf-8"):
    """
    Input CSV header: edge_id, from, to, flow
    - Builds new vertex ids 0..n-1 (order of first appearance)
    - Writes new CSV with remapped from/to
    Returns: (n_nodes, tail, head, mapping)
    """
    in_csv, out_csv = Path(in_csv), Path(out_csv)

    vid = {}          # old_vertex -> new_id
    tail, head = [], []

    def get_id(v: str) -> int:
        v = v.strip()
        if v not in vid:
            vid[v] = len(vid)
        return vid[v]

    with in_csv.open("r", newline="", encoding=encoding) as fin, \
         out_csv.open("w", newline="", encoding=encoding) as fout:

        reader = csv.DictReader(fin, delimiter=';')
        writer = csv.writer(fout)
        writer.writerow(["edge_id", "from", "to", "flow"])

        for r in reader:
            u = get_id(r["from"])
            v = get_id(r["to"])
            tail.append(u)
            head.append(v)
            writer.writerow([r["edge_id"], u, v, r["flow"]])

    tail = np.asarray(tail, dtype=np.int32)
    head = np.asarray(head, dtype=np.int32)
    n_nodes = len(vid)

    return n_nodes, tail, head, vid

if __name__ == "__main__":
    n, tail, head, mapping = remap_edges_csv("./edges.csv", "./edges_remapped.csv")
    G = beckmann.CSRGraph.from_edges(n, tail, head)
    g_nx = G.to_networkx()
    nx.draw(g_nx, with_labels=False, node_size=300, arrows=True)
    plt.show()
