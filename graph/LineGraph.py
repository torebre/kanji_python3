
import networkx as nx
from import_data import LineData, LinePositionMap, ClusterLookupMap

class LineGraph:

    # def __init__(self, line_map: LinePositionMap, cluster_lookup_map: ClusterLookupMap):
    #     self.line_graph = nx.Graph()
        # for entry in line_map.items():
        #     print(entry[0])


    def __init__(self, relation_data):
        self.line_graph = nx.Graph()

        for row in relation_data:
            self.line_graph.add_node(row[0])
            self.line_graph.add_node(row[1])

            self.line_graph.add_edge(row[0], row[1])



