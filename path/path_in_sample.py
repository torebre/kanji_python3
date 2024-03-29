from typing import List

from path.path_element import PathElement


class Path:
    sample_id: int
    path: List[PathElement]

    def __init__(self, sample_id: int, path_steps=None):
        if path_steps is None:
            path_steps = []
        self.sample_id = sample_id
        self.path = path_steps

    def get_element_ids(self) -> List[int]:
        return [sample.id_to_within_sample for sample in self.path]

    def get_distance(self) -> float:
        total_distance = 0
        for step in self.path:
            total_distance += step.distance

        return 1 / (total_distance + 0.001) + len(self.path)
