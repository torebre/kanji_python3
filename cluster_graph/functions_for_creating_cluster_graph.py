from typing import Set, List, Dict, ValuesView

import networkx as nx
import numpy as np

from cluster_graph.LineClustering import LineClustering
from clustering.dbscan_clustering_test import LineCodeMap, extract_line_code_map_from_array, do_dbscan, \
    add_label_data_to_line_code_map, extract_cluster_relation_data, find_relation_sets_for_all_last_four_lines
from import_data import import_data
from import_data.RelationData import RelationData
from import_data.import_data import IntegerToListOfIntegerMap


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


def create_cluster_using_last_four_lines(line_data: List[Dict]) -> LineClustering:
    line_data = line_data

    # The last four lines are the ones that make up a rectangle
    last_four_lines: IntegerToListOfIntegerMap = import_data.filter_out_four_last_lines_of_data(line_data)

    array_data = import_data.transform_selected_lines_to_array(line_data, last_four_lines)
    line_code_line_id_relation_data_map: LineCodeMap = extract_line_code_map_from_array(array_data)

    # data_used_for_clustering = array_data[:, 3:6]
    data_used_for_clustering = array_data[:, 5]

    # Trying with only angle to see if clusters look as expected
    # data_used_for_clustering = array_data[:, 4:6]

    # dbscan_test(data_used_for_clustering.reshape(-1, 1))
    # plot_dbscan(data_used_for_clustering)

    # dbscan_data = do_dbscan(data_used_for_clustering)
    dbscan_data = do_dbscan(np.reshape(data_used_for_clustering, (-1, 1)))
    add_label_data_to_line_code_map(dbscan_data.labels_, array_data, line_code_line_id_relation_data_map)

    distinct_labels = set(dbscan_data.labels_)

    extract_cluster_relation_data(line_code_line_id_relation_data_map, last_four_lines)

    relation_sets: Dict[int, ValuesView[RelationData]] = find_relation_sets_for_all_last_four_lines(
        last_four_lines,
        line_code_line_id_relation_data_map)

    # TODO Update constructor
    return LineClustering(line_data, last_four_lines, distinct_labels, relation_sets)


def create_cluster_using_all_lines(line_data: List[Dict]):
    line_map: IntegerToListOfIntegerMap = import_data.setup_line_data_map(line_data)
    array_data = import_data.transform_selected_lines_to_array(line_data, line_map)
    line_code_line_id_relation_data_map: LineCodeMap = extract_line_code_map_from_array(array_data)

    data_used_for_clustering = array_data[:, 4:6]

    # dbscan_test(data_used_for_clustering.reshape(-1, 1))
    # plot_dbscan(data_used_for_clustering)

    dbscan_data = do_dbscan(data_used_for_clustering)
    # dbscan_data = do_dbscan(np.reshape(data_used_for_clustering, (-1, 1)))
    add_label_data_to_line_code_map(dbscan_data.labels_, array_data, line_code_line_id_relation_data_map)

    distinct_labels = set(dbscan_data.labels_)

    extract_cluster_relation_data(line_code_line_id_relation_data_map, line_map)

    relation_sets: Dict[int, ValuesView[RelationData]] = find_relation_sets_for_all_last_four_lines(
        line_map,
        line_code_line_id_relation_data_map)

    return LineClustering(line_data, distinct_labels, relation_sets)
