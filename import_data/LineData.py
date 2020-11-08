from typing import List
from RelativePositionData import RelativePositionData


class LineData:
    line_id: int
    relation_data: List[RelativePositionData]

    def __init__(self, line_id: int, relation_data: List[RelativePositionData]):
        self.line_id = line_id
        self.relation_data = relation_data
