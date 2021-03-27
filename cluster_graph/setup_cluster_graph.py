import networkx as nx

from cluster_graph.LineClustering import LineClustering
from cluster_graph.functions_for_creating_cluster_graph import setup_base_cluster_graph, \
    add_data_from_single_image_graph_to_cluster_graph, add_data_from_single_image_graph_to_cluster_graph2
from cluster_graph.functions_for_creating_graph_for_single_image import create_cluster_from_relation_data
from import_data import import_data
import matplotlib.pyplot as plt

from typing import Dict, ValuesView

from import_data.RelationData import RelationData


def setup_graph_type_1() -> nx.MultiDiGraph:
    line_data = import_data.read_relation_data()
    line_cluster = LineClustering(line_data)
    relation_sets:  Dict[int, ValuesView[RelationData]] = line_cluster.relation_sets

    cluster_graph: nx.MultiDiGraph = setup_base_cluster_graph(line_cluster.distinct_labels)

    for relation_set_id in relation_sets:
        line_graph = create_cluster_from_relation_data(relation_sets[relation_set_id])
        add_data_from_single_image_graph_to_cluster_graph(line_graph, cluster_graph)

    return cluster_graph


def setup_graph_type_2() -> nx.MultiDiGraph:
    line_data = import_data.read_relation_data()
    line_cluster = LineClustering(line_data)
    relation_sets:  Dict[int, ValuesView[RelationData]] = line_cluster.relation_sets

    cluster_graph: nx.MultiDiGraph = nx.MultiDiGraph()

    for relation_set_id in relation_sets:
        line_graph = create_cluster_from_relation_data(relation_sets[relation_set_id])
        add_data_from_single_image_graph_to_cluster_graph2(relation_set_id, line_graph, cluster_graph)

    return cluster_graph


if __name__ == '__main__':
    cluster_graph: nx.MultiDiGraph = setup_graph_type_2()

    for node in cluster_graph.nodes:
        print(node)

    # nx.write_gexf(cluster_graph, 'test_cluster_graph_export2.gexf')
    nx.write_graphml(cluster_graph, 'test_cluster_graph_export2.graphml')

    # nx.draw(cluster_graph)
    # plt.draw()

