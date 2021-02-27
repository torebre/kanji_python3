import matplotlib.pyplot as plt
import sys

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from clustering.dbscan_clustering_test import LineCodeMap, extract_line_code_map_from_array, do_dbscan, \
    add_label_data_to_line_code_map, extract_cluster_relation_data, find_relation_sets_for_all_last_four_lines, \
    generate_color_map_for_line
from clustering.hierarchical_clustering import hierarchical_cluster_test
from import_data import import_data
import numpy as np
import networkx as nx

from import_data.RelationData import RelationData
from import_data.import_data import LastFourLinesMap
from typing import Dict, Tuple, List, ValuesView



def create_cluster_from_relation_data(relation_sets: Dict[int, ValuesView[RelationData]]):
    line_graph = nx.Graph()

    # TODO It does not make sense to add data from different images to the same graph
    for image_id in relation_sets:
        relation_data = relation_sets[image_id]

        for entry in relation_data:
            line_graph.add_node(entry.line_id)
            for other_node_id in entry.other_lines_cluster_map:
                line_graph.add_edge(entry.line_id, other_node_id, cluster=entry.other_lines_cluster_map[other_node_id])

    nx.write_gexf(line_graph, 'test_graph_export.gexf')


def setup_relation_data():
    line_data = import_data.read_relation_data()

    # line_code_line_id_relation_data_map = import_data.transform_to_line_code_map(line_data)

    # The last four lines are the ones that make up a rectangle
    last_four_lines: LastFourLinesMap = import_data.filter_out_four_last_lines_of_data(line_data)

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
    # plot_dbscan(dbscan_data, data_used_for_clustering) #np.reshape(data_used_for_clustering, (len(data_used_for_clustering), 1)))

    distinct_labels = set(dbscan_data.labels_)

    extract_cluster_relation_data(line_code_line_id_relation_data_map, last_four_lines)

    relation_sets: Dict[int, ValuesView[RelationData]] = find_relation_sets_for_all_last_four_lines(last_four_lines,
                                                                                                    line_code_line_id_relation_data_map)
    for key in relation_sets:
        print(key, ":")
        for relation_data in relation_sets[key]:
            print(relation_data)

    return relation_sets


if __name__ == '__main__':
    relation_sets = setup_relation_data()
    create_cluster_from_relation_data(relation_sets)
