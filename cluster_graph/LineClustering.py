from typing import Dict, ValuesView
import numpy as np

from cluster_graph.functions_for_creating_graph_for_single_image import create_cluster_from_relation_data
from clustering.dbscan_clustering_test import LineCodeMap, extract_line_code_map_from_array, do_dbscan, \
    add_label_data_to_line_code_map, extract_cluster_relation_data, find_relation_sets_for_all_last_four_lines
from import_data import import_data
from import_data.RelationData import RelationData
from import_data.import_data import LastFourLinesMap



class LineClustering:

    def __init__(self, line_data):
        self.line_data = line_data

        # The last four lines are the ones that make up a rectangle
        self.last_four_lines: LastFourLinesMap = import_data.filter_out_four_last_lines_of_data(self.line_data)

        array_data = import_data.transform_selected_lines_to_array(self.line_data, self.last_four_lines)
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

        self.distinct_labels = set(dbscan_data.labels_)

        extract_cluster_relation_data(line_code_line_id_relation_data_map, self.last_four_lines)

        self.relation_sets: Dict[int, ValuesView[RelationData]] = find_relation_sets_for_all_last_four_lines(self.last_four_lines,
                                                                                                        line_code_line_id_relation_data_map)
        # for key in relation_sets:
        #     print(key, ":")
        #     for relation_data in relation_sets[key]:
        #         print(relation_data)
