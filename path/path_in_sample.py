from typing import List

from path.path_element import PathElement


class Path:
    sample_id: int
    path: List[PathElement] = []

    def __init__(self, sample_id: int):
        self.sample_id = sample_id

    def extend(self, id_within_sample_of_new_step: int, distance: float):
        self.path.append(PathElement(id_within_sample_of_new_step, distance))

    def get_element_ids(self) -> List[int]:
        return [sample.id_to_within_sample for sample in self.path]
