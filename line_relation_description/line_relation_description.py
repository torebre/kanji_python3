import math
import random
from typing import List, Dict, Tuple, Set

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

from line_data_generation.generate_training_sample import generate_training_samples
from line_data_generation.line_relation_data_setup import extract_closest_neighbours_for_line
from path.similar_samples import SimilarSamples
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
                               number_of_closest_lines_to_return: int = 10) -> Tuple[npt.NDArray, npt.NDArray]:
    '''
    Returns a tuple where the first array contains the indices of the closest lines, and the
    second array contains the distances.
    '''
    # TODO Using midpoint diff will not work for finding rectangles
    angle_diffs = abs(angle_diff - _data[:, 0])
    sorted_angle_diffs_indices = angle_diffs.argsort()

    # Return the indices of the smallest angle differences
    return (sorted_angle_diffs_indices[0:number_of_closest_lines_to_return],
            angle_diffs[sorted_angle_diffs_indices[0:number_of_closest_lines_to_return]])


def find_similar_paths(test_sample: npt.ArrayLike, data: npt.NDArray,
                       index_of_start_line: int = None) -> Dict[
    Tuple[int, int], Set[Tuple[int, int]]]:
    '''

    :param test_sample:
    :param index_of_start_line
    :param data:
    :return: The key is the IDs of the first and second steps in the input sample. One step consists of two lines.
    The value is a tuple consisting of the index of the first step among the lookup samples, index of the second step
    among the lookup samples, distance between first step of the input and lookup sample, distance between the second
    step from the input sample and the second step among the lookup samples
    '''
    # test_sample = np.sort(test_sample, axis=1)
    # Order the rows representing the lines by length
    # test_sample = test_sample[test_sample[:, 1].argsort()[::-1]]
    number_of_closest_neighbours_to_return = 5

    if index_of_start_line is None:
        index_first_line = len(test_sample) - 1
    else:
        index_first_line = index_of_start_line

    closest_neighbours = extract_closest_neighbours_for_line(index_first_line, test_sample,
                                                             number_of_closest_neighbours_to_return)
    input_line = test_sample[index_first_line]
    input_similar_map: Dict[Tuple[int, int], Set[Tuple[int, int, int, int]]] = {}
    for second_line_in_path in closest_neighbours:
        (angle_diff, midpoint_x_diff, midpoint_y_diff) = describe_two_lines(input_line,
                                                                            test_sample[second_line_in_path])
        (row_indices_of_closest_lines_across_lookup_examples, first_distances) = find_closest_lines_in_data(angle_diff,
                                                                                                            midpoint_x_diff,
                                                                                                            midpoint_y_diff,
                                                                                                            data)
        closest_neighbours_for_second_line = extract_closest_neighbours_for_line(second_line_in_path, test_sample,
                                                                                 number_of_closest_neighbours_to_return)

        sample_indices = {}
        counter = 0
        for index in row_indices_of_closest_lines_across_lookup_examples:
            row = data[index]
            sample_indices[(row[3], row[5])] = (index, first_distances[counter])
            counter += 1

        # sample_indices = {(row[3], row[5]) for row in data[indices_of_closest_lines_across_lookup_examples]}

        second_line_data = test_sample[second_line_in_path]
        for third_line_in_path in closest_neighbours_for_second_line:
            similar_line_configurations: Set[Tuple[int, int, int, int]] = set()

            if third_line_in_path == index_first_line or third_line_in_path == second_line_in_path:
                # Do not go back and look at the first line
                continue

            (angle_diff, midpoint_x_diff, midpoint_y_diff) = describe_two_lines(second_line_data,
                                                                                test_sample[third_line_in_path])
            (indices_of_closest_lines_across_lookup_examples2, second_distances) = find_closest_lines_in_data(
                angle_diff,
                midpoint_x_diff,
                midpoint_y_diff,
                data)

            counter2 = 0
            for index in indices_of_closest_lines_across_lookup_examples2:
                row = data[index]
                key = (row[3], row[4])
                # sample_indices2[(row[3], row[5])] = index

                if key in sample_indices:
                    # If there is a pair of lines in the first step where the second line is
                    # the same as the first line in the second step, then add it here
                    # to add one step to the path started in the first step
                    index_of_first_line, distance_for_first_step = sample_indices[key]
                    similar_line_configurations.add((
                        index_of_first_line, index, distance_for_first_step, second_distances[counter2]))

                counter2 += 1

            if len(similar_line_configurations) != 0:
                input_similar_map[(second_line_in_path, third_line_in_path)] = similar_line_configurations

                # for key in sample_indices2:
                #     if key in sample_indices:
                #         similar_line_configurations.add((sample_indices[key], sample_indices2[key]))

                # intersection = sample_indices.intersection(sample_indices2)

                # print(f"Intersection: {intersection}")
    # print("Similar line configurations: ", similar_line_configurations)

    return input_similar_map


def find_similar_paths2(input_sample_id: int, test_sample: npt.ArrayLike, data: npt.NDArray,
                        indices_of_lines_to_use: List[int]) -> SimilarSamples:
    '''

    :param test_sample:
    :param indices_of_lines_to_use
    :param data: An array where the columns are angle difference, midpoint x difference, midpoint y difference,
        sample index, line 1 index within sample and line 2 index within sample
    :return:
    '''
    current_index = indices_of_lines_to_use[0]
    # paths: Dict[int, List[List[Tuple[int, int, int, float]]]] = {}
    similar_samples: SimilarSamples = SimilarSamples(input_sample_id, test_sample, indices_of_lines_to_use)

    similar_samples.input_sample_id = input_sample_id

    for line_index in indices_of_lines_to_use[1:]:
        # Describe the relation between two lines
        first_line_in_relation = test_sample[current_index]
        second_line_in_relation = test_sample[line_index]
        (angle_diff, midpoint_x_diff, midpoint_y_diff) = describe_two_lines(first_line_in_relation,
                                                                            second_line_in_relation)
        # Look for similar relations between lines in the data set
        (row_indices_of_closest_lines_across_lookup_examples, first_distances) = find_closest_lines_in_data(angle_diff,
                                                                                                            midpoint_x_diff,
                                                                                                            midpoint_y_diff,
                                                                                                            data)
        counter = 0
        for index in row_indices_of_closest_lines_across_lookup_examples:
            row = data[index]
            # sample_indices[(row[3], row[5])] = (index, first_distances[counter])

            paths_to_extend = similar_samples.find_paths_where_last_step_is_matching(row[3], row[4])

            if len(paths_to_extend) == 0:
                similar_samples.start_new_path(int(row[3]), int(row[4]), int(row[5]), first_distances[counter])
            else:
                for path in paths_to_extend:
                    path.extend(int(row[4]), first_distances[counter])

            counter += 1

        return similar_samples

    # for index in indices_of_closest_lines_across_lookup_examples2:
    #     row = data[index]
    #     key = (row[3], row[4])
    #     # sample_indices2[(row[3], row[5])] = index
    #
    #     if key in sample_indices:
    #         # If there is a pair of lines in the first step where the second line is
    #         # the same as the first line in the second step, then add it here
    #         # to add one step to the path started in the first step
    #         index_of_first_line, distance_for_first_step = sample_indices[key]
    #         similar_line_configurations.add((
    #             index_of_first_line, index, distance_for_first_step, second_distances[counter2]))
    #
    #     counter2 += 1
    #
    # if len(similar_line_configurations) != 0:
    #     input_similar_map[(second_line_in_path, third_line_in_path)] = similar_line_configurations

    # return input_similar_map


def show_results(data, index_first_line, input_similar_map, samples_in_lookup,
                 test_sample, number_of_closest_samples_to_show: int = 30):
    for key in input_similar_map:
        similar_line_configurations = input_similar_map[key]

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

        similar_line_configurations_sorted_by_distance = sorted(similar_line_configurations,
                                                                key=lambda value: value[2] + value[3])
        configuration_counter = 0
        for similar_configuration in similar_line_configurations_sorted_by_distance:
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

            plt.matshow(line_matrix)
            plt.title(
                f"Sample: {sample_index}. Line pairs: {similar_configuration[0]}, {similar_configuration[1]}.\nDistances: {similar_configuration[2]}, {similar_configuration[3]}")
            plt.show()

            configuration_counter += 1

            if configuration_counter >= number_of_closest_samples_to_show:
                break


def show_results2(data, similar_sample: SimilarSamples, samples_in_lookup: List[npt.ArrayLike]):
    # Show the input
    line_coordinates = generate_line_coordinates_from_matrix(similar_sample.input_sample)
    line_values = []
    counter2 = 0

    for counter in range(len(line_coordinates)):
        if counter in similar_sample.ids_of_input_elements:
            line_values.append(100 + counter2)
            counter2 += 100
        else:
            line_values.append(10)

    line_matrix = get_line_matrix(line_coordinates, line_values)

    plt.matshow(line_matrix)
    plt.title(f"Input ID: {similar_sample.input_sample_id}")
    plt.show()

    for similar_path in similar_sample.similar_paths_in_other_samples:
        print(f"Sample ID: {similar_path.sample_id}")

        lookup_sample = samples_in_lookup[similar_path.sample_id]
        indices_in_sample_list = []

        for path_element in similar_path.path:
            indices_in_sample_list.append(path_element.id_to_within_sample)

        line_coordinates = generate_line_coordinates_from_matrix(lookup_sample)
        line_values = []

        counter2 = 0
        for counter in range(len(line_coordinates)):
            if counter in indices_in_sample_list:
                line_values.append(100 + counter2)
                counter2 += 100
            else:
                line_values.append(10)

        line_matrix = get_line_matrix(line_coordinates, line_values)

        plt.matshow(line_matrix)
        plt.title(
            f"Sample: {similar_path.sample_id}. Line indices: {similar_path.get_element_ids()}")
        plt.show()


# def search_for_rectangle_experiment_with_one_random_line():
#     random.seed(1)
#     # Try to include just one random line in the cases where there is a rectangle in the sample
#     all_training_samples = generate_training_samples(100, 95, 1)
#     test_sample = all_training_samples[0]
#     index_first_line = len(test_sample) - 1
#
#     # data has all the lines for all the samples except the sample that is going to be used for testing lookup
#     # data = setup_example_rows(range(1, len(training_samples)), training_samples)
#
#     # Only include one sample to look for rectangles for debugging purposes
#     samples_to_include_in_lookup = [1]
#     data: npt.NDArray = setup_example_rows(samples_to_include_in_lookup, all_training_samples)
#     samples_in_lookup = [all_training_samples[1]]
#
#     input_similar_map = find_similar_paths(test_sample, data)
#     # input_similar_map_sorted = {key: value for key, value in sorted(input_similar_map.items(), key = lambda item: item[2] + item[3])}
#
#     show_results(data, index_first_line, input_similar_map, samples_in_lookup, test_sample)


def search_for_rectangle_experiment_with_multiple_random_lines():
    random.seed(2)

    all_training_samples = generate_training_samples(100, 50, 10)
    test_sample = all_training_samples[0]
    index_first_line = len(test_sample) - 1

    # data has all the lines for all the samples except the sample that is going to be used for testing lookup
    # data = setup_example_rows(range(1, len(training_samples)), training_samples)

    samples_to_include_in_lookup = [i for i in range(1, len(all_training_samples))]
    data: npt.NDArray = setup_example_rows(samples_to_include_in_lookup, all_training_samples)
    samples_in_lookup = [all_training_samples[i] for i in samples_to_include_in_lookup]

    indices_of_lines_to_use = [0, 1, 2, 3]
    input_similar_map = find_similar_paths2(0, test_sample, data, indices_of_lines_to_use)
    # input_similar_map_sorted = {key: value for key, value in sorted(input_similar_map.items(), key = lambda item: item[2] + item[3])}

    show_results(data, index_first_line, input_similar_map, samples_in_lookup, test_sample)


def search_for_rectangle_experiment_with_multiple_random_lines_rectangle_lines_as_input():
    random.seed(2)

    all_training_samples = generate_training_samples(100, 50, 10)
    test_sample = all_training_samples[0]

    samples_to_include_in_lookup = [i for i in range(1, len(all_training_samples))]
    data: npt.NDArray = setup_example_rows(samples_to_include_in_lookup, all_training_samples)
    samples_in_lookup: List[npt.ArrayLike] = [all_training_samples[i] for i in samples_to_include_in_lookup]

    number_of_lines_in_test_sample = test_sample.shape[0]
    indices_of_lines_to_use = [number_of_lines_in_test_sample - 1,
                               number_of_lines_in_test_sample - 2,
                               number_of_lines_in_test_sample - 3,
                               number_of_lines_in_test_sample - 4]
    closest_paths: SimilarSamples = find_similar_paths2(0, test_sample, data, indices_of_lines_to_use)

    closest_paths.sort_paths_sorted_by_distance_criteria()
    show_results2(data, closest_paths, samples_in_lookup)


if __name__ == "__main__":
    # search_for_rectangle_experiment_with_one_random_line()
    # search_for_rectangle_experiment_with_multiple_random_lines()
    search_for_rectangle_experiment_with_multiple_random_lines_rectangle_lines_as_input()
