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

    def find_paths_where_last_step_is_matching(self, sample_id: int, id_last_path_element: int, id_of_new_step: int) -> \
            List[
                path_in_sample.Path]:
        matching_paths = []

        for path in self.similar_paths_in_other_samples:
            if sample_id != path.sample_id:
                continue

            step_already_in_path = False
            for path_step in path.path:
                if path_step.id_to_within_sample == id_of_new_step:
                    step_already_in_path = True
                    break

            if step_already_in_path:
                continue

            for path_step in path.path[:-1]:
                if path_step.id_to_within_sample == id_last_path_element:
                    step_already_in_path = True
                    break

            if step_already_in_path:
                continue

            if path.path[-1].id_to_within_sample == id_last_path_element:
                matching_paths.append(path)

        return matching_paths

    def start_new_path(self, sample_id: int, id_of_first_element_within_sample: int,
                       id_of_second_element_within_sample: int, distance: float):
        new_path = path_in_sample.Path(sample_id)
        new_path.path.append(path_element.PathElement(id_of_first_element_within_sample, 0.0))
        new_path.path.append(path_element.PathElement(id_of_second_element_within_sample, distance))

        self.similar_paths_in_other_samples.append(new_path)

    def sort_path_by_score_ascending(self):

        def sorting_function(path_to_get_distance_for: Path):
            return path_to_get_distance_for.get_distance()

        self.similar_paths_in_other_samples = sorted(self.similar_paths_in_other_samples, key=sorting_function,
                                                     reverse=True)

    def add_path_with_one_more_element(self, path_to_extend: Path, id_within_sample_of_new_step: int, distance: float):
        new_path = []
        for path_step in path_to_extend.path:
            new_path.append(PathElement(path_step.id_to_within_sample, path_step.distance))
        new_path.append(PathElement(id_within_sample_of_new_step, distance))

        self.similar_paths_in_other_samples.append(Path(path_to_extend.sample_id, new_path))

    def exists_path_starting_with_id(self, id_within_sample_of_new_step: int) -> bool:
        for similar_path in self.similar_paths_in_other_samples:
            if similar_path.path[0].id_to_within_sample == id_within_sample_of_new_step:
                return True

        return False
