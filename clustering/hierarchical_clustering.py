import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage

from import_data import import_data


def hierarchical_cluster_test(data):
    # linked = linkage(data, 'single')
    linked = linkage(data, method='single', metric='cityblock')

    dendrogram(linked,
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
    hierarchical_cluster_test(data[0:1000, :])
