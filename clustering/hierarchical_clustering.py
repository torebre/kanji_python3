import matplotlib.pyplot as plt
import sys

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from import_data import import_data


def hierarchical_cluster_test(data):
    return linkage(data, method='single', metric='cityblock')


def plot_dendrogram(linkage_data):
    dendrogram(linkage_data,
                      orientation='top',
                      labels=range(0,data.shape[0]),
                      distance_sort='descending',
                      show_leaf_counts=True)
    plt.figure(figsize=(10, 7))
    plt.show()


if __name__ == '__main__':
    line_data = import_data.read_data()
    data = import_data.transform_to_array(line_data)
    # Use subset of data because of problems running on the full data set
    # hierarchical_cluster_test(data[0:1000, :])

    sys.setrecursionlimit(100000)
    linkage_data = hierarchical_cluster_test(data)
    plot_dendrogram(linkage_data)

    flat_cluster = fcluster(linkage_data, 1)
    number_of_clusters = len(set(flat_cluster))
    print("Number of clusters:", number_of_clusters)
