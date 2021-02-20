class RelationData:
    line_id: int
    other_lines_cluster_map: dict

    def __init__(self, line_id: int, other_lines_cluster_map=None):
        if other_lines_cluster_map is None:
            other_lines_cluster_map = {}
        self.line_id = line_id
        self.other_lines_cluster_map = other_lines_cluster_map

    def __str__(self):
        return "ID: " + str(self.line_id) + ". Relations: " + str(self.other_lines_cluster_map)
