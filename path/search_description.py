from typing import List

import numpy.typing as npt


class SearchStep:
    test_sample: npt.ArrayLike
    first_line_in_relation: npt.ArrayLike
    second_line_in_relation: npt.ArrayLike


class SearchDescription:
    search_steps: List[SearchStep]
