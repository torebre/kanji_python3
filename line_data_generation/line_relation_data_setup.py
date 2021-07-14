from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from line_data_generation.generate_training_sample import generate_training_samples
from line_utilities.create_line import create_line


def extract_closest_neighbours_for_line(line_number: int, lines_matrix: npt.ArrayLike,
                                        max_number_of_lines_to_return=5) -> \
        List[int]:
    distance_matrix = generate_distance_matrix(line_number, lines_matrix)
    max_value = np.iinfo(np.int32).max
    line_min_distance_tuple_list = []

    row_count = -1
    for row in lines_matrix:
        row_count += 1
        if row_count == line_number:
            continue

        line = create_line(row[2].astype(np.int32),
                           row[3].astype(np.int32),
                           row[4].astype(np.int32),
                           row[5].astype(np.int32))

        min_distance_for_line = max_value
        for coordinate in line:
            distance_for_point = distance_matrix[coordinate[0]][coordinate[1]]
            if min_distance_for_line > distance_for_point:
                min_distance_for_line = distance_for_point

        line_min_distance_tuple_list.append((row_count, min_distance_for_line))

    line_min_distance_tuple_list = sorted(line_min_distance_tuple_list, key=lambda tup: tup[1])

    return [tup[0] for tup in line_min_distance_tuple_list[0:max_number_of_lines_to_return]]


def generate_distance_matrix(line_number: int, lines_matrix: npt.ArrayLike) -> npt.ArrayLike:
    distance_matrix = np.zeros(
        (max(np.concatenate((lines_matrix[:, 2], lines_matrix[:, 4]), axis=0)).astype(np.int32) + 1,
         max(np.concatenate((lines_matrix[:, 3], lines_matrix[:, 5]), axis=0)).astype(np.int32) + 1),
        dtype=np.int32)
    distance_matrix.fill(-1)

    line = create_line(lines_matrix[line_number][2].astype(np.int32),
                       lines_matrix[line_number][3].astype(np.int32),
                       lines_matrix[line_number][4].astype(np.int32),
                       lines_matrix[line_number][5].astype(np.int32))

    coordinates_to_examine = []
    for coordinate_pair in line:
        distance_matrix[coordinate_pair[0]][coordinate_pair[1]] = 0
        coordinates_to_examine.append(coordinate_pair)

    distance_to_fill_in = 1

    coordinates_to_examine_next = []

    while True:
        while len(coordinates_to_examine) > 0:
            for coordinate in get_neighbours_with_no_distance_set(coordinates_to_examine[0][0],
                                                                  coordinates_to_examine[0][1],
                                                                  distance_matrix):
                if coordinate not in coordinates_to_examine_next:
                    coordinates_to_examine_next.append(coordinate)
            coordinates_to_examine.pop(0)

        if len(coordinates_to_examine_next) == 0:
            break

        fill_in_neighbours(coordinates_to_examine_next, distance_to_fill_in, distance_matrix)

        coordinates_to_examine = coordinates_to_examine_next
        coordinates_to_examine_next = []
        distance_to_fill_in += 1

    return distance_matrix


def fill_in_neighbours(coordinates: List[Tuple[int]], distance_to_fill_in: int, distance_matrix: npt.ArrayLike):
    for coordinate in coordinates:
        distance_matrix[coordinate[0]][coordinate[1]] = distance_to_fill_in

    # positions_to_update = [[i, j] if 0 <= i < len(distance_matrix) and 0 <= j < len(distance_matrix[0]) and not (
    #         i == x_coord and j == y_coord) and distance_matrix[i][j] == -1 else None
    #                        for j in range(x_coord - 2, x_coord + 1)
    #                        for i in range(y_coord - 2, y_coord + 1)]
    #
    # for position_to_update in positions_to_update:
    #     if position_to_update is None:
    #         continue
    #
    #     distance_matrix[position_to_update[0]][positions_to_update[1]] = distance_to_fill_in


def get_neighbours_with_no_distance_set(x_coord: int, y_coord: int, distance_matrix: npt.ArrayLike):
    positions_to_update = [
        (i, j) if 0 <= i < len(distance_matrix) and 0 <= j < len(distance_matrix[0]) and distance_matrix[i][
            j] == -1 else None
        for j in range(y_coord - 1, y_coord + 2)
        for i in range(x_coord - 1, x_coord + 2)]

    filtered_list = list(filter(lambda coordinate_pair: coordinate_pair is not None, positions_to_update))

    return filtered_list


def add_line_to_matrix(start_x: int, start_y: int, stop_x: int, stop_y: int, matrix_data: npt.ArrayLike,
                       fill_value: int):
    line = create_line(start_x, start_y, stop_x, stop_y)

    for coordinate_pair in line:
        matrix_data[coordinate_pair[0]][coordinate_pair[1]] = fill_value


if __name__ == '__main__':
    training_samples = generate_training_samples()
    temp_matrix = generate_distance_matrix(0, training_samples[0])
    closest_neighbours = extract_closest_neighbours_for_line(0, training_samples[0])

    print("Closest neighbours: ", closest_neighbours)

    line_counter = 0
    for training_sample in training_samples[0]:
        if line_counter == 0:
            fill_value = -100
        elif line_counter in closest_neighbours:
            fill_value = -50
        else:
            fill_value = -3

        add_line_to_matrix(training_sample[2].astype(np.int32), training_sample[3].astype(np.int32),
                           training_sample[4].astype(np.int32), training_sample[5].astype(np.int32), temp_matrix,
                           fill_value)

        line_counter += 1

    plt.imshow(temp_matrix)
    plt.show()
