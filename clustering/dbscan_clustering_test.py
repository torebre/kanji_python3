from import_data import import_data

import numpy as np
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt


def dbscan_test(distance_data):
    db = DBSCAN(eps=+.3, min_samples=10).fit(distance_data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

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

    plt.show()


if __name__ == '__main__':
    line_data = import_data.read_data()
    last_four_lines = import_data.filter_out_four_last_lines_of_data(line_data)

    data = import_data.transform_selected_lines_to_array(line_data, last_four_lines)

    # data_used_for_clustering = data[:, 3:6]

    # Trying with only angle to see if clusters look as expected
    data_used_for_clustering = data[:, 4:6]

    # dbscan_test(data_used_for_clustering.reshape(-1, 1))
    dbscan_test(data_used_for_clustering)
