import os
import numpy as np
import pandas as pd
from src.od_matrix_completion.core.models.beckmann_solver import BeckmannSolver

def load_example_data():
    net_file = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/master/SiouxFalls/SiouxFalls_net.tntp"
    demand_file = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/master/SiouxFalls/CSV-data/SiouxFalls_od.csv"
    # OD matrix
    dem = pd.read_csv(demand_file)
    zones = int(max(dem.O.max(), dem.D.max()))
    index = np.arange(zones) + 1
    mtx = np.zeros(shape=(zones, zones))
    for element in dem.to_records(index=False):
        mtx[element[0]-1][element[1]-1] = element[2]
    # Network itsels
    net = pd.read_csv(net_file, skiprows=2, sep="\t", lineterminator=";", header=None)
    net.columns = ["newline", "a_node", "b_node", "capacity", "length", "free_flow_time", "b", "power", "speed", "toll", "link_type", "terminator"]
    net.drop(columns=["newline", "terminator"], index=[76], inplace=True)
    network = net[['a_node', 'b_node', "capacity", 'free_flow_time']]
    network = network.assign(direction=1)
    network["link_id"] = network.index + 1
    network = network.astype({"a_node":"int64", "b_node": "int64"})
    return network, mtx, index

def main():
    # Я хочу получить:
    # 1) links -- это рёбра и их харакретистики в терминах задачи
    # 2) zones -- это названия зон, для которых считается OD матрица
    # 3) D -- это сама OD матрица (самое простое, что может быть тут)
    links, D, zones = load_example_data()

    solver = BeckmannSolver(links, zones)
    solver.setup_assignment(D)
    print(solver.links_)

    solver.solve()

    res = solver.results()
    print(res)


if __name__ == "__main__":
    main()
