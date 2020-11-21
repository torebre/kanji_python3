from import_data.LineData import LineData
from import_data import LinePositionMap


class RelationDataExtraction:
    def __extract_relation_data(self, line: LineData, line_map: LinePositionMap):
        # TODO
        print("Test")

    def extract_relation_data_for_all_lines(self, line_map: LinePositionMap):
        for line in line_map:
            self.__extract_relation_data(line_map[line], line_map)
