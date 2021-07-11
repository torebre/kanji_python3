from typing import Dict
import pandas as pd
import numpy as np

import networkx as nx

from visualize.create_line_svg import draw_line_data_on_svg_canvas

from import_data import import_data
from line_data_generation.generate_training_sample import generate_training_samples

# def setup_graph_type_2_with_all_lines() -> nx.MultiDiGraph:
#     line_data = import_data.read_relation_data()
#     line_cluster = create_cluster_using_all_lines(line_data)
#     relation_sets: Dict[int, ValuesView[RelationData]] = line_cluster.relation_sets
#
#     cluster_graph: nx.MultiDiGraph = nx.MultiDiGraph()
#
#     for relation_set_id in relation_sets:
#         line_graph = create_cluster_from_relation_data(relation_sets[relation_set_id])
#         add_data_from_single_image_graph_to_cluster_graph2(relation_set_id, line_graph, cluster_graph)
#
#     return cluster_graph


# def create_and_print_cluster_graph():
#     cluster_graph: nx.MultiDiGraph = setup_graph_type_2_with_all_lines()
#
#     for node in cluster_graph.nodes:
#         for neighbour in cluster_graph.neighbors(node):
#             print('From: ', node, '. To: ', neighbour)
#
#             edge_data = cluster_graph.get_edge_data(node, neighbour)
#             number_of_edges = len(edge_data)
#
#             print("Number of edges: ", number_of_edges)


if __name__ == '__main__':
    training_samples = generate_training_samples()

    rows_in_sample = len(training_samples[0])

    print("Rows in sample: ", rows_in_sample)

    transformed_sample = np.zeros((rows_in_sample, 6))
    for i in range(rows_in_sample):
        transformed_sample[i, 0] = 1
        transformed_sample[i, 1] = i
        transformed_sample[i, 2:6] = training_samples[0][i, 0:4]

    dataframe = pd.DataFrame(transformed_sample,
                             columns=["unicode", "line_number", "angle", "length", "start_x", "start_y"])

    canvas = draw_line_data_on_svg_canvas(dataframe)
    canvas.setPixelScale(5)
    canvas.savePng('test_output_svg2.png')
