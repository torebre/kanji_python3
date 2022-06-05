from typing import List

import numpy.typing as npt
from path import path_element, path_in_sample
from path.path_element import PathElement
from path.path_in_sample import Path


class SimilarSamples:
    input_sample_id: int
    input_sample: npt.ArrayLike
    ids_of_input_elements: List[int]
    similar_paths_in_other_samples: List[path_in_sample.Path] = []

    def __init__(self, input_sample_id: int, input_sample: npt.ArrayLike, ids_of_input_elements: List[int]):
        self.input_sample_id = input_sample_id
        self.input_sample = input_sample
        self.ids_of_input_elements = ids_of_input_elements

    def find_paths_where_last_step_is_matching(self, sample_id: int, id_last_path_element: int) -> List[
        path_in_sample.Path]:
        matching_paths = []

        for path in self.similar_paths_in_other_samples:
            if sample_id != path.sample_id:
                continue

            if path.path[-1] == id_last_path_element:
                matching_paths.append(path)

        return matching_paths

    def start_new_path(self, sample_id: int, id_of_first_element_within_sample: int,
                       id_of_second_element_within_sample: int, distance: float):
        new_path = path_in_sample.Path(sample_id)
        new_path.path.append(path_element.PathElement(id_of_first_element_within_sample, 0.0))
        new_path.path.append(path_element.PathElement(id_of_second_element_within_sample, distance))

        self.similar_paths_in_other_samples.append(new_path)

    def sort_paths_sorted_by_distance_criteria(self):

        def sorting_function(path_to_get_distance_for: Path):
            total_distance = 0
            for step in path_to_get_distance_for.path:
                total_distance += step.distance

            return total_distance / len(path_to_get_distance_for.path)

        self.similar_paths_in_other_samples = sorted(self.similar_paths_in_other_samples, key=sorting_function)

    def add_path_with_one_more_element(self, path_to_extend: Path, id_within_sample_of_new_step: int, distance: float):
        new_path = []
        for path_step in path_to_extend.path:
            new_path.append(PathElement(path_step.id_to_within_sample, path_step.distance))
        new_path.append(PathElement(id_within_sample_of_new_step, distance))

        self.similar_paths_in_other_samples.append(new_path)
