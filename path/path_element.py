class PathElement:
    id_to_within_sample: int
    # Distance from the previous step to this one
    distance: float

    def __init__(self, id_to_within_sample: int, distance: float):
        self.id_to_within_sample = id_to_within_sample
        self.distance = distance
