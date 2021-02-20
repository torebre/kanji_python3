from import_data import import_data

import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

from import_data.RelationData import RelationData
from import_data.import_data import LastFourLinesMap

# The map has the line_code as key. The second key is (line code, input line, other line). The values are (row difference, column difference and angle)
LineCodeMap = Dict[int, Dict[Tuple[int, int, int], Tuple[int, int, float]]]

LineLineColourMap = Dict[int, Dict[int, int]]


def do_dbscan(distance_data):
    return DBSCAN(eps=+.3, min_samples=10).fit(distance_data)


def plot_dbscan(dbscan_data, distance_data):
    core_samples_mask = np.zeros_like(dbscan_data.labels_, dtype=bool)
    core_samples_mask[dbscan_data.core_sample_indices_] = True
    labels = dbscan_data.labels_

    unique_labels = set(labels)
    colours = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colours):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = distance_data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)

        xy = distance_data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    plt.show()


def add_label_data_to_line_code_map(dbscan_labels, array_data, line_code_map):
    index = 0
    for row in array_data:
        new_tuple = line_code_map[row[0]][(row[0], row[1], row[2])] + (dbscan_labels[index],)
        line_code_map[row[0]][(row[0], row[1], row[2])] = new_tuple
        index += 1


def extract_line_code_map_from_array(array_data) -> LineCodeMap:
    """The columns assumed to be in the array are: lineCode, inputLine, otherLine, rowDiff, colDiff and angle."""

    line_code_map = {}
    for row in array_data:
        line_code = row[0]

        if line_code in line_code_map:
            line_code_map[line_code][(line_code, row[1], row[2])] = (row[3], row[4], row[5])
        else:
            line_code_map[line_code] = {(line_code, row[1], row[2]): (row[3], row[4], row[5])}

    return line_code_map


def extract_cluster_relation_data(line_code_map, last_four_lines_id_map):
    line_code_relation_last_four_lines_relation_matrix_map = {}
    for line_code, relation_data in line_code_map.items():
        line_code_relation_last_four_lines_relation_matrix_map[
            line_code] = extract_rectangle_relation_data_for_line_code(line_code, line_code_map, last_four_lines_id_map)

    return line_code_relation_last_four_lines_relation_matrix_map


def extract_rectangle_relation_data_for_line_code(line_code, line_code_map, last_four_lines_id_map):
    relation_data = line_code_map[line_code]
    cluster_relation_matrix = np.full((4, 4), np.nan)

    line_id_index_map = {}

    # print("Line code: ", line_code)
    for key, value in enumerate(last_four_lines_id_map[line_code]):
        line_id_index_map[value] = key

    for tuple_id, tuple_values in relation_data.items():
        input_line = line_id_index_map[tuple_id[1]]
        other_line = line_id_index_map[tuple_id[2]]
        cluster_relation_matrix[input_line, other_line] = tuple_values[3]

        # print("Input line: ", input_line, ". Other line: ", other_line, ". Values: ", tuple_values[0], tuple_values[1],
        #       tuple_values[2])

    # print("Cluster relation matrix for ", line_code)
    # print(cluster_relation_matrix)

    return cluster_relation_matrix


def get_sets_of_relation_data_for_last_four_lines(line_code, line_code_map: LineCodeMap,
                                                  last_four_lines_id_map: LastFourLinesMap):
    relation_data = line_code_map[line_code]

    # Four lines, put the cluster type of the relations
    # to the other lines into four sets
    relation_sets = [list(), list(), list(), list()]
    line_id_index_map = {}

    relation_data_list = []
    for key, value in enumerate(last_four_lines_id_map[line_code]):
        line_id_index_map[value] = RelationData(value)

    for tuple_id, tuple_values in relation_data.items():
        relation_data = line_id_index_map[tuple_id[1]]

        relation_data.other_lines_cluster_map[tuple_id[2]] = tuple_values[3]
        # relation_sets[key].append(tuple_values[3])

    return line_id_index_map.values()
    # return relation_sets


def find_relation_sets_for_all_last_four_lines(last_four_lines: LastFourLinesMap,
                                               line_code_line_id_relation_data_map: LineCodeMap):
    line_code_relation_sets_map = {}
    for key in last_four_lines:
        relation_sets_for_line_code = get_sets_of_relation_data_for_last_four_lines(key,
                                                                                    line_code_line_id_relation_data_map,
                                                                                    last_four_lines)
        line_code_relation_sets_map[key] = relation_sets_for_line_code

    return line_code_relation_sets_map


def generate_color_map_for_line(relation_data: RelationData, cluster_color_map: dict) -> LineLineColourMap:
    color_map = {}

    for key in relation_data.other_lines_cluster_map:
        cluster_id = relation_data.other_lines_cluster_map[key]
        color_map[key] = cluster_color_map[cluster_id]

    return color_map


def create_line_colour_map_for_line_code(line_code: int, line_code_relation_sets_map: dict, cluster_colour_map: dict) -> dict:
    line_to_line_colour_map = {}

    for relation_data in line_code_relation_sets_map[line_code]:
        line_line_colour_map = generate_color_map_for_line(relation_data, cluster_colour_map)

        for first_line_id in line_line_colour_map:
            line_colours = line_line_colour_map[first_line_id]
            line_to_line_colour_map[(relation_data.line_id, first_line_id)] = line_colours

    return line_to_line_colour_map


def create_cluster_colour_map(distinct_labels: list) -> dict:
    number_of_clusters = len(distinct_labels)
    cluster_colour_map = {}
    counter = 0
    diff = 255 / number_of_clusters

    for label in distinct_labels:
        cluster_colour_map[label] = '#%02x%02x%02x' % (int(counter), int(counter), 0)
        counter += diff

    return cluster_colour_map


if __name__ == '__main__':
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

    # first_line_code = next(iter(line_code_line_id_relation_data_map))
    # relation_sets_for_line_code = get_sets_of_relation_data_for_last_four_lines(first_line_code,
    #                                                                             line_code_line_id_relation_data_map,
    #                                                                             last_four_lines)
    # print("Relation sets: ", relation_sets_for_line_code)

    line_to_line_colour_map = {}

    relation_sets = find_relation_sets_for_all_last_four_lines(last_four_lines, line_code_line_id_relation_data_map)
    # for key in relation_sets:
    #     print(key, ":")
    #     for relation_data in relation_sets[key]:
    #         print(relation_data)
    #
    #         line_line_colour_map = generate_color_map_for_line(relation_data, cluster_colour_map)
    #
    #         # print("Line colour map:", line_line_colour_map)
    #
    #         for first_line_id in line_line_colour_map:
    #             line_colours = line_line_colour_map[first_line_id]
    #             line_to_line_colour_map[(key, first_line_id)] = line_colours
    #
    # print("Line to line colour map: ", line_to_line_colour_map)

    # colour_map = create_line_colour_map(relation_sets)

    # print("Colour map:", colour_map)
