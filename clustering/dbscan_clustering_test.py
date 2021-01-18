from import_data import import_data

import numpy as np
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt


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


def extract_line_code_map_from_array(array_data):
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
    for line_code, relation_data in line_code_map.items():
        extract_rectangle_relation_data_for_line_code(line_code, line_code_map, last_four_lines_id_map)


def extract_rectangle_relation_data_for_line_code(line_code, line_code_map, last_four_lines_id_map):
    relation_data = line_code_map[line_code]
    cluster_relation_matrix = np.full((4, 4), np.nan)

    line_id_index_map = {}

    print("Line code: ", line_code)
    for key, value in enumerate(last_four_lines_id_map[line_code]):
        line_id_index_map[value] = key

    for tuple_id, tuple_values in relation_data.items():
        input_line = line_id_index_map[tuple_id[1]]
        other_line = line_id_index_map[tuple_id[2]]
        cluster_relation_matrix[input_line, other_line] = tuple_values[3]

        print("Input line: ", input_line, ". Other line: ", other_line, ". Values: ", tuple_values[0], tuple_values[1],
              tuple_values[2])

    print("Cluster relation matrix for ", line_code)
    print(cluster_relation_matrix)


if __name__ == '__main__':
    line_data = import_data.read_relation_data()

    # line_code_line_id_relation_data_map = import_data.transform_to_line_code_map(line_data)

    # The last four lines are the ones that make up a rectangle
    last_four_lines = import_data.filter_out_four_last_lines_of_data(line_data)

    array_data = import_data.transform_selected_lines_to_array(line_data, last_four_lines)
    line_code_line_id_relation_data_map = extract_line_code_map_from_array(array_data)

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

    extract_cluster_relation_data(line_code_line_id_relation_data_map, last_four_lines)
