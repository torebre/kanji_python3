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


def add_data_from_single_image_graph_to_cluster_graph2(line_code: int, line_graph: nx.DiGraph, graph: nx.MultiDiGraph):
    for node in line_graph.adj:
        add_path_step(line_code, node, '1', 1, line_graph, graph, set())


def add_path_step(line_code: int, current_node, from_node_label: str, step_count: int, line_graph: nx.DiGraph, graph: nx.MultiDiGraph,
                  taboo_list: Set[int]):
    if step_count == 5:
        return

    for neighbour in nx.neighbors(line_graph, current_node):
        if neighbour in taboo_list:
            continue

        edge = line_graph.get_edge_data(current_node, neighbour)
        cluster_number = edge['cluster']
        to_node_label = str(step_count + 1) + '_' + str(cluster_number)
        graph.add_edge(from_node_label, to_node_label, line_code=line_code,
                       from_node=current_node, to_node=neighbour, cluster=cluster_number)
        taboo_list.add(neighbour)

        add_path_step(line_code, neighbour, to_node_label, step_count + 1, line_graph, graph, taboo_list)


def find_paths(start_node_label: str, current_edge: dict, graph: nx.MultiDiGraph):


    for neighbour in graph.neighbors(start_node_label):
        node_attributes = nx.get_node_attributes(graph, neighbour)

        edge_data = graph.get_edge_data(start_node_label, neighbour)


        # TODO




