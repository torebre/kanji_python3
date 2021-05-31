import networkx as nx
from cluster_graph.LineClustering import LineClustering
from cluster_graph.functions_for_creating_cluster_graph import setup_base_cluster_graph, \
    add_data_from_single_image_graph_to_cluster_graph, add_data_from_single_image_graph_to_cluster_graph2, \
    create_graph_showing_number_of_paths, create_cluster_using_last_four_lines, create_cluster_using_all_lines
from cluster_graph.functions_for_creating_graph_for_single_image import create_cluster_from_relation_data
from import_data import import_data
import matplotlib.pyplot as plt

from typing import Dict, ValuesView

from import_data.RelationData import RelationData


def setup_graph_type_1() -> nx.MultiDiGraph:
    line_data = import_data.read_relation_data()
    line_cluster = LineClustering(line_data)
    relation_sets: Dict[int, ValuesView[RelationData]] = line_cluster.relation_sets

    cluster_graph: nx.MultiDiGraph = setup_base_cluster_graph(line_cluster.distinct_labels)

    for relation_set_id in relation_sets:
        line_graph = create_cluster_from_relation_data(relation_sets[relation_set_id])
        add_data_from_single_image_graph_to_cluster_graph(line_graph, cluster_graph)

    return cluster_graph


def setup_graph_type_2() -> nx.MultiDiGraph:
    line_data = import_data.read_relation_data()
    line_cluster = create_cluster_using_last_four_lines(line_data)
    relation_sets: Dict[int, ValuesView[RelationData]] = line_cluster.relation_sets

    cluster_graph: nx.MultiDiGraph = nx.MultiDiGraph()

    for relation_set_id in relation_sets:
        line_graph = create_cluster_from_relation_data(relation_sets[relation_set_id])
        add_data_from_single_image_graph_to_cluster_graph2(relation_set_id, line_graph, cluster_graph)

    return cluster_graph


def setup_graph_type_2_with_all_lines() -> nx.MultiDiGraph:
    line_data = import_data.read_relation_data()
    line_cluster = create_cluster_using_all_lines(line_data)
    relation_sets: Dict[int, ValuesView[RelationData]] = line_cluster.relation_sets

    cluster_graph: nx.MultiDiGraph = nx.MultiDiGraph()

    for relation_set_id in relation_sets:
        line_graph = create_cluster_from_relation_data(relation_sets[relation_set_id])
        add_data_from_single_image_graph_to_cluster_graph2(relation_set_id, line_graph, cluster_graph)

    return cluster_graph


def create_and_print_cluster_graph():
    # cluster_graph: nx.MultiDiGraph = setup_graph_type_2()
    cluster_graph: nx.MultiDiGraph = setup_graph_type_2_with_all_lines()

    for node in cluster_graph.nodes:
        for neighbour in cluster_graph.neighbors(node):
            print('From: ', node, '. To: ', neighbour)

            edge_data = cluster_graph.get_edge_data(node, neighbour)
            number_of_edges = len(edge_data)

            print("Number of edges: ", number_of_edges)

            # for key in edge_data:
            #     print("Edge: ", edge_data[key])

            # print(edge_data)

    # nx.write_gexf(cluster_graph, 'test_cluster_graph_export2.gexf')
    # nx.write_graphml(cluster_graph, 'test_cluster_graph_export2.graphml')

    # nx.draw(cluster_graph)
    # plt.draw()


def create_and_show_graph_with_number_of_edges():
    cluster_graph: nx.MultiDiGraph = setup_graph_type_2()
    edge_graph: nx.DiGraph = create_graph_showing_number_of_paths('1', cluster_graph)

    # nx.write_gexf(cluster_graph, 'test_cluster_graph_export2.gexf')
    # nx.write_graphml(edge_graph, 'graph_with_number_of_edges.graphml')

    nx.draw(edge_graph)
    plt.draw()
    plt.show()


if __name__ == '__main__':
    # create_and_show_graph_with_number_of_edges()
    create_and_print_cluster_graph()