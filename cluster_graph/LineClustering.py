from typing import Dict, ValuesView, List

from import_data.RelationData import RelationData


class LineClustering:
    line_data: List[Dict]
    # last_four_lines: IntegerToListOfIntegerMap
    distinct_labels: set
    relation_sets: Dict[int, ValuesView[RelationData]]

    def __init__(self, line_data: List[Dict],
                 # last_four_lines: IntegerToListOfIntegerMap,
                 distinct_labels: set,
                 relation_sets: Dict[int, ValuesView[RelationData]]):
        self.line_data = line_data
        # self.last_four_lines = last_four_lines
        self.distinct_labels = distinct_labels
        self.relation_sets = relation_sets
