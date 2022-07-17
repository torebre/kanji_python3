from typing import List

import numpy.typing as npt

from path.path_in_sample import Path
from path.similar_samples import SimilarSamples


class SearchStep:
    first_line_in_relation: npt.ArrayLike
    second_line_in_relation: npt.ArrayLike

    row_indices_of_closest_lines_across_lookup_examples: npt.NDArray
    first_distances: npt.NDArray

    paths_to_extend: List[Path]

    similar_samples: SimilarSamples

class SearchDescription:
    test_sample: npt.ArrayLike
    search_steps: List[SearchStep] = []
