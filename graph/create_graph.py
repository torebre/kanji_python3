from import_data import import_data, RelationDataExtraction
import sys
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from clustering import hierarchical_clustering as hc
from graph import LineGraph


if __name__ == '__main__':
    line_data = import_data.read_data()

    line_relation_data = import_data.transform_to_array(line_data)

    line_graph = LineGraph.LineGraph(line_relation_data)

    # line_data_map = import_data.generate_line_data(line_data)
    #
    # data = import_data.transform_to_array(line_data)
    #
    # sys.setrecursionlimit(100000)
    # linkage_data = hc.hierarchical_cluster_test(data)
    # # plot_dendrogram(linkage_data)
    #
    # flat_cluster = fcluster(linkage_data, 1)
    # number_of_clusters = len(set(flat_cluster))
    # print("Number of clusters:", number_of_clusters)
