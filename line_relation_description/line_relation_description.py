import math
from typing import Iterable, List

import numpy as np
import numpy.typing as npt

from line_data_generation.generate_training_sample import generate_training_samples
from line_data_generation.line_relation_data_setup import extract_closest_neighbours_for_line


def describe_two_lines(line1: npt.ArrayLike, line2: npt.ArrayLike):
    """
        Angle, line length, start x coordinate, start y coordinate, stop x coordinate, stop y coordinate
    :return:
    """
    line_angle1 = compute_line_angle(line1[1], line1[2], line1[3], line1[4])
    line_angle2 = compute_line_angle(line2[1], line2[2], line2[3], line2[4])

    angle_diff = abs(line_angle1 - line_angle2)

    midpoint_x_line1 = line1[1] if line1[1] < line1[3] else line1[3] + abs(line1[1] - line1[3]) / 2
    midpoint_x_line2 = line2[1] if line2[1] < line2[3] else line2[3] + abs(line2[1] - line2[3]) / 2

    midpoint_y_line1 = line1[2] if line1[2] < line1[4] else line1[4] + abs(line1[2] - line1[4]) / 2
    midpoint_y_line2 = line2[3] if line2[2] < line2[4] else line2[4] + abs(line2[2] - line2[4]) / 2

    return (angle_diff, abs(midpoint_x_line1 - midpoint_x_line2), abs(midpoint_y_line1 - midpoint_y_line2))


def compute_line_angle(start_x: int, start_y: int, stop_x: int, stop_y: int):
    x_delta = stop_x - start_x
    y_delta = stop_y - start_y

    # TODO Check that this is the correct function to use
    return math.atan2(y_delta, x_delta)


# def compute_line_length(start_x: int, start_y: int, stop_x: int, stop_y: int):
#     x_delta = stop_x - start_x
#     y_delta = stop_y - start_y
#
#     return math.sqrt(x_delta * x_delta + y_delta * y_delta)


def setup_example_rows(indices_lookup_examples: List[int], input_data) -> npt.NDArray:
    """
        Returns an array where the columns are angle difference, midpoint x difference, midpoint y difference, sample index, line 1 index within sample and line 2 index within sample
    """
    lookup_examples = [input_data[index] for index in indices_lookup_examples]
    number_of_lookup_examples = len(lookup_examples)
    # There are the same number of lines in all the samples
    number_of_lines_in_examples = len(input_data[0])
    # dataframe = pd.DataFrame(data=np.zeros(shape=(8, number_of_lookup_examples * (number_of_lookup_examples - 1))),
    # columns=["unicode", "line_number", "angle", "length", "start_x", "start_y", "index_line1",
    #          "index_line2"])
    _data = np.zeros((number_of_lookup_examples * number_of_lines_in_examples * (number_of_lines_in_examples - 1), 6))
    example_counter = 0
    sample_index = 0
    for sample in lookup_examples:
        _count = 0

        for _line in sample:
            _count2 = 0

            for _line2 in sample:
                if _count == _count2:
                    _count2 += 1
                    continue

                (_angle_diff, _midpoint_x_diff, _midpoint_y_diff) = describe_two_lines(_line, _line2)
                _data[example_counter] = [_angle_diff, _midpoint_x_diff, _midpoint_y_diff, sample_index, _count,
                                          _count2]
                example_counter += 1
                _count2 += 1

                # print(f"{angle_diff}, {midpoint_x_diff}, {midpoint_y_diff}")

            _count += 1

        sample_index += 1

    print(example_counter)

    return _data


def find_closest_lines_in_data(angle_diff, midpoint_x_diff, midpoint_y_diff, _data) -> npt.NDArray:
    # TODO Using midpoint diff will not work for finding rectangles

    # angle_diffs = np.concatenate(range(0, data.shape[0]), abs(angle_diff - data[:, 0]))

    angle_diffs = abs(angle_diff - _data[:, 0])
    sorted_angle_diffs_indices = angle_diffs.argsort()

    # print(sorted_angle_diffs_indices)

    # Return the indices of the 100 smallest angle differences
    return sorted_angle_diffs_indices[0:100]


if __name__ == "__main__":
    training_samples = generate_training_samples(100, 5)
    test_sample = training_samples[0]

    # data has all the lines for all the samples except the sample that is going to be used for testing lookup
    data = setup_example_rows(range(1, len(training_samples)), training_samples)

    # test_sample = np.sort(test_sample, axis=1)

    # Order the rows representing the lines by length
    test_sample = test_sample[test_sample[:, 1].argsort()[::-1]]

    closest_neighbours = extract_closest_neighbours_for_line(0, test_sample)

    print(closest_neighbours)

    input_line = test_sample[0]
    for row_number in closest_neighbours:
        (angle_diff, midpoint_x_diff, midpoint_y_diff) = describe_two_lines(input_line, test_sample[row_number])
        indices_of_closest_lines_across_lookup_examples = find_closest_lines_in_data(angle_diff, midpoint_x_diff,
                                                                                     midpoint_y_diff, data)
        closest_neighbours_for_nearby_line = extract_closest_neighbours_for_line(row_number, test_sample)

        sample_indices = set(data[indices_of_closest_lines_across_lookup_examples][:, 3])

        for nearby_line_index in closest_neighbours_for_nearby_line:
            closest_neighbours2 = extract_closest_neighbours_for_line(nearby_line_index, test_sample)

            for row_number2 in closest_neighbours2:
                if row_number2 == 0:
                    # Line 0 is the original input line, do not go back and look at it
                    continue

                input_line2 = test_sample[row_number]

                (angle_diff, midpoint_x_diff, midpoint_y_diff) = describe_two_lines(input_line2, test_sample[row_number2])
                indices_of_closest_lines_across_lookup_examples2 = find_closest_lines_in_data(angle_diff,
                                                                                              midpoint_x_diff,
                                                                                              midpoint_y_diff,
                                                                                              data)
                sample_indices2 = set(data[indices_of_closest_lines_across_lookup_examples2][:, 3])

                intersection = sample_indices.intersection(sample_indices2)

                print(f"Intersection: {intersection}")


    # count = 0
    # for line in sample:
    #     count2 = 0
    #
    #     for line2 in sample:
    #         if count == count2:
    #             count2 += 1
    #             continue
    #
    #         (angle_diff, midpoint_x_diff, midpoint_y_diff) = describe_two_lines(line, line2)

    # Find closest points in example lookup set

    # Choose new line pair

    # Find closest points for new line pair
