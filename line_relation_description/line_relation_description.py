import math
import random
from typing import Iterable, List, Dict, Tuple, Set

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

from line_data_generation.generate_training_sample import generate_training_samples
from line_data_generation.line_relation_data_setup import extract_closest_neighbours_for_line
from line_utilities.create_line import get_line_matrix
from visualize.DrawLines import generate_line_coordinates_from_matrix


def describe_two_lines(line1: npt.ArrayLike, line2: npt.ArrayLike):
    """
        Angle, line length, start x coordinate, start y coordinate, stop x coordinate, stop y coordinate
    :return:
    """
    # line_angle1 = compute_line_angle(line1[1], line1[2], line1[3], line1[4])
    # line_angle2 = compute_line_angle(line2[1], line2[2], line2[3], line2[4])
    # angle_diff = abs(line_angle1 - line_angle2)

    angle_diff = abs(line1[0] - line2[0])
    if angle_diff > 2 * math.pi:
        angle_diff -= 2 * math.pi
    elif angle_diff < 0:
        angle_diff += 2 * math.pi


    midpoint_x_line1 = line1[1] if line1[1] < line1[3] else line1[3] + abs(line1[1] - line1[3]) / 2
    midpoint_x_line2 = line2[1] if line2[1] < line2[3] else line2[3] + abs(line2[1] - line2[3]) / 2

    midpoint_y_line1 = line1[2] if line1[2] < line1[4] else line1[4] + abs(line1[2] - line1[4]) / 2
    midpoint_y_line2 = line2[3] if line2[2] < line2[4] else line2[4] + abs(line2[2] - line2[4]) / 2

    return (angle_diff, abs(midpoint_x_line1 - midpoint_x_line2), abs(midpoint_y_line1 - midpoint_y_line2))


def describe_two_lines_updated(angle_line1: float, angle_line2: float, line1_x_start: int, line1_y_start: int,
                       line1_x_stop: int, line1_y_stop: int, line2_x_start: int, line2_y_start: int,
                       line2_x_stop: int, line2_y_stop: int):
    """
    The two lines are assumed to come from the same sample

    :return:
    """
    angle_diff = abs(angle_line1 - angle_line2)
    if angle_diff > 2 * math.pi:
        angle_diff -= 2 * math.pi
    elif angle_diff < 0:
        angle_diff += 2 * math.pi

    midpoint_x_line1 = (line1_x_start if line1_x_start < line1_x_stop else line1_x_stop) + abs(
        line1_x_start - line1_x_stop) / 2
    midpoint_x_line2 = (line2_x_start if line2_x_start < line2_x_stop else line2_x_stop) + abs(
        line2_x_start - line2_x_stop) / 2

    midpoint_y_line1 = (line1_y_start if line1_y_start < line1_y_stop else line1_y_stop) + abs(
        line1_y_start - line1_y_stop) / 2
    midpoint_y_line2 = (line2_y_start if line2_y_start < line2_y_stop else line2_y_stop) + abs(
        line2_y_start - line2_y_stop) / 2

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
        Returns an array where the columns are angle difference, midpoint x difference, midpoint y difference,
        sample index, line 1 index within sample and line 2 index within sample
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


def find_closest_lines_in_data(angle_diff, midpoint_x_diff, midpoint_y_diff, _data,
                               number_of_closest_lines_to_return: int = 10) -> npt.NDArray:
    # TODO Using midpoint diff will not work for finding rectangles

    # angle_diffs = np.concatenate(range(0, data.shape[0]), abs(angle_diff - data[:, 0]))

    angle_diffs = abs(angle_diff - _data[:, 0])
    sorted_angle_diffs_indices = angle_diffs.argsort()

    # print(sorted_angle_diffs_indices)

    # Return the indices of the smallest angle differences
    return sorted_angle_diffs_indices[0:number_of_closest_lines_to_return]


if __name__ == "__main__":
    random.seed(1)
    # Try to include just one random line in the cases where there is a rectangle in the sample
    all_training_samples = generate_training_samples(100, 95, 1)
    test_sample = all_training_samples[0]

    # data has all the lines for all the samples except the sample that is going to be used for testing lookup
    # data = setup_example_rows(range(1, len(training_samples)), training_samples)

    # Only include one sample to look for rectangles for debugging purposes
    samples_to_include_in_lookup = [1]
    data = setup_example_rows(samples_to_include_in_lookup, all_training_samples)
    samples_in_lookup = [all_training_samples[1]]

    # test_sample = np.sort(test_sample, axis=1)

    # Order the rows representing the lines by length
    # test_sample = test_sample[test_sample[:, 1].argsort()[::-1]]

    index_first_line = len(test_sample) - 1
    number_of_closest_neighbours_to_return = 5
    closest_neighbours = extract_closest_neighbours_for_line(index_first_line, test_sample,
                                                             number_of_closest_neighbours_to_return)
    input_line = test_sample[index_first_line]

    input_similar_map: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}

    for second_line_in_path in closest_neighbours:
        (angle_diff, midpoint_x_diff, midpoint_y_diff) = describe_two_lines(input_line,
                                                                            test_sample[second_line_in_path])
        row_indices_of_closest_lines_across_lookup_examples = find_closest_lines_in_data(angle_diff, midpoint_x_diff,
                                                                                         midpoint_y_diff, data)
        closest_neighbours_for_second_line = extract_closest_neighbours_for_line(second_line_in_path, test_sample,
                                                                                 number_of_closest_neighbours_to_return)

        sample_indices = {}
        for index in row_indices_of_closest_lines_across_lookup_examples:
            row = data[index]
            sample_indices[(row[3], row[5])] = index

        # sample_indices = {(row[3], row[5]) for row in data[indices_of_closest_lines_across_lookup_examples]}

        second_line_data = test_sample[second_line_in_path]
        for third_line_in_path in closest_neighbours_for_second_line:
            similar_line_configurations: Set[Tuple[int, int]] = set()

            if third_line_in_path == index_first_line or third_line_in_path == second_line_in_path:
                # Do not go back and look at the first line
                continue

            (angle_diff, midpoint_x_diff, midpoint_y_diff) = describe_two_lines(second_line_data,
                                                                                test_sample[third_line_in_path])
            indices_of_closest_lines_across_lookup_examples2 = find_closest_lines_in_data(angle_diff,
                                                                                          midpoint_x_diff,
                                                                                          midpoint_y_diff,
                                                                                          data)

            # sample_indices2 = {}
            for index in indices_of_closest_lines_across_lookup_examples2:
                row = data[index]
                key = (row[3], row[4])
                # sample_indices2[(row[3], row[5])] = index

                if key in sample_indices:
                    # If there is a pair of lines in the first step where the second line is
                    # the same as the first line in the second step, then add it here
                    # to add one step to path started in the first step
                    similar_line_configurations.add((sample_indices[key], index))

            if len(similar_line_configurations) != 0:
                input_similar_map[(second_line_in_path, third_line_in_path)] = similar_line_configurations

                # for key in sample_indices2:
                #     if key in sample_indices:
                #         similar_line_configurations.add((sample_indices[key], sample_indices2[key]))

                # intersection = sample_indices.intersection(sample_indices2)

                # print(f"Intersection: {intersection}")

    # print("Similar line configurations: ", similar_line_configurations)

    for key in input_similar_map:
        similar_line_configurations = input_similar_map[key]

        print(f"Input lines: {key}")

        # Show the input
        line_coordinates = generate_line_coordinates_from_matrix(test_sample)
        line_values = []
        counter2 = 0

        for counter in range(len(line_coordinates)):
            if counter == index_first_line or counter in key:
                line_values.append(100 + counter2)
                counter2 += 100
            else:
                line_values.append(10)

        line_matrix = get_line_matrix(line_coordinates, line_values)

        # fig = plt.figure()

        plt.matshow(line_matrix)
        plt.title(f"Input: {index_first_line}, {key[0]}, {key[1]}")

        # plt.text(line_coordinates[0][0][0], line_coordinates[0][0][1], "Test")
        # plt.text(0, 0, "Test")

        plt.show()

        for similar_configuration in similar_line_configurations:
            sample_index = data[similar_configuration[0]][3]

            print("Sample index: ", sample_index)

            lookup_sample = samples_in_lookup[sample_index.astype(int)]

            indices_in_sample = [data[similar_configuration[0]][4],
                                 data[similar_configuration[0]][5],
                                 data[similar_configuration[1]][5]]

            line_coordinates = generate_line_coordinates_from_matrix(lookup_sample)
            line_values = []

            counter2 = 0
            for counter in range(len(line_coordinates)):
                if counter in indices_in_sample:
                    line_values.append(100 + counter2)
                    counter2 += 100
                else:
                    line_values.append(10)

            line_matrix = get_line_matrix(line_coordinates, line_values)

            # fig = plt.figure()

            plt.matshow(line_matrix)
            plt.title(f"Sample: {sample_index}. Lines: ")
            plt.show()
