import networkx as nx

from cluster_graph.LineClustering import LineClustering
from cluster_graph.functions_for_creating_graph_for_single_image import create_cluster_from_relation_data
from import_data import import_data

if __name__ == '__main__':
    line_data = import_data.read_relation_data()
    line_cluster = LineClustering(line_data)
    relation_sets = line_cluster.relation_sets
    line_graph = create_cluster_from_relation_data(relation_sets[next(iter(relation_sets))])
    nx.write_gexf(line_graph, 'test_graph_export.gexf')
