from typing import Set

import networkx as nx


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


def add_path_step(line_code: int, current_node, from_node_label: str, step_count: int, line_graph: nx.DiGraph,
                  graph: nx.MultiDiGraph,
                  taboo_list: Set[int]):
    if step_count == 5:
        return

    for neighbour in nx.neighbors(line_graph, current_node):
        if neighbour in taboo_list:
            continue

        edge = line_graph.get_edge_data(current_node, neighbour)
        cluster_number = edge['cluster']
        to_node_label = str(step_count + 1) + '_' + str(cluster_number)

        existing_edges = graph.get_edge_data(from_node_label, to_node_label, default=None)
        if not edge_exists(existing_edges, line_code, current_node, neighbour):
            graph.add_edge(from_node_label, to_node_label, line_code=line_code,
                           from_node=current_node, to_node=neighbour, cluster=cluster_number)
            taboo_list.add(neighbour)
            add_path_step(line_code, neighbour, to_node_label, step_count + 1, line_graph, graph, taboo_list)


def edge_exists(existing_edges: dict, line_code, current_node, neighbour) -> bool:
    if existing_edges is not None:
        for existing_edge in existing_edges:
            existing_edge_data = existing_edges[existing_edge]

            if existing_edge_data['line_code'] == line_code and existing_edge_data['from_node'] == current_node and \
                    existing_edge_data['to_node'] == neighbour:
                # Only include the edge once per line code
                return True

    return False


def create_graph_showing_number_of_paths(start_node_label: str,
                                         cluster_graph: nx.MultiDiGraph) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(start_node_label)
    process_neighbours(start_node_label, graph, cluster_graph)

    return graph


def process_neighbours(start_node_label: str, graph: nx.DiGraph, cluster_graph: nx.MultiDiGraph):
    for neighbour in cluster_graph.neighbors(start_node_label):
        node_attributes = nx.get_node_attributes(cluster_graph, neighbour)
        edge_data = cluster_graph.get_edge_data(start_node_label, neighbour)

        print("Neighbour:", neighbour)
        print("Node attributes:", node_attributes)
        print("Edge data:", len(edge_data))

        graph.add_node(neighbour)

        # Create an edge with an attribute saying how many
        # edges there are between the nodes in the multigraph
        graph.add_edge(start_node_label, neighbour, number_of_edges=len(edge_data))

        process_neighbours(neighbour, graph, cluster_graph)
