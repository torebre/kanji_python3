from import_data import LineData


def extract_relation_data(line: LineData, line_map: dict[int, LineData]):
    for relation_data in line.relation_data:
        # TODO
        print(relation_data)
