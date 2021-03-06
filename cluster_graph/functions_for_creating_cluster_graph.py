import networkx as nx

from typing import Set


def setup_base_cluster_graph(cluster_ids: Set[int]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()

    for cluster_id in cluster_ids:
        graph.add_node(cluster_id)

    return graph


def add_data_from_single_image_graph_to_cluster_graph(line_graph: nx.DiGraph, graph: nx.MultiDiGraph):
    for node in line_graph.adj:
        for neighbour in nx.neighbors(line_graph, node):
            edge = line_graph.get_edge_data(node, neighbour)
            for neighbour2 in nx.neighbors(line_graph, neighbour):
                edge2 = line_graph.get_edge_data(neighbour, neighbour2)
                graph.add_edge(edge['cluster'], edge2['cluster'], key=None, nodes_from=str(node) + "_" + str(neighbour),
                               nodes_to=str(neighbour) + "_" + str(neighbour2))
