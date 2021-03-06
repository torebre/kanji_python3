import networkx as nx

from import_data.RelationData import RelationData
from typing import Dict, Tuple, List, ValuesView


def create_cluster_from_relation_data(relation_data: ValuesView[RelationData]) -> nx.Graph:
    line_graph = nx.Graph()

    for entry in relation_data:
        line_graph.add_node(entry.line_id)
        for other_node_id in entry.other_lines_cluster_map:
            line_graph.add_edge(entry.line_id, other_node_id, cluster=entry.other_lines_cluster_map[other_node_id])

    return line_graph
