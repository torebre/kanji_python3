# from typing import TypedDict
from typing_extensions import TypedDict

from import_data.LineData import LineData


class LinePositionMap(TypedDict):
    line_id: int
    line_data: LineData


class RelationDataExtraction:

    def __extract_relation_data(self, line: LineData, line_map: LinePositionMap):
        for relation_data in line.relation_data:
            # TODO
            print(relation_data)

    def extract_relation_data_for_all_lines(self, line_map: LinePositionMap):
        for line in line_map:
            self.__extract_relation_data(line_map[line], line_map)

