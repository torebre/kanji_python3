import json
import numpy as np
import matplotlib.pyplot as plt

from import_data.RelativePositionData import RelativePositionData
from import_data.LineData import LineData


def read_data():
    with open(
            '/home/student/workspace/testEncodings/line_relative_position_information_rectangle_v2.json') as input_data:
        data = json.load(input_data)
        return data


def transform_to_line_code_map(line_data):
    line_code_map = {}

    for line in line_data:
        line_code = line["lineCode"]

        for relative_position in line['relativePositions']:
            line_code_map[(line_code, relative_position['inputLine'], relative_position['otherLine'])] = (
            relative_position['rowDiff'], relative_position['colDiff'], relative_position['angle'])

    return line_code_map


def transform_to_array(data):
    """The columns in the returned matrix are: lineCode, inputLine, otherLine, rowDiff, colDiff and angle."""
    total_data_counter = 0
    for line in data:
        for _ in line['relativePositions']:
            total_data_counter += 1

    number_of_variables = 6
    position_data = np.zeros(total_data_counter * number_of_variables).reshape(total_data_counter, number_of_variables)
    counter = 0

    for line in data:
        line_code = line["lineCode"]
        for relative_position in line['relativePositions']:
            position_data[counter, 0] = line_code
            position_data[counter, 1] = relative_position['inputLine']
            position_data[counter, 2] = relative_position['otherLine']
            position_data[counter, 3] = relative_position['rowDiff']
            position_data[counter, 4] = relative_position['colDiff']
            position_data[counter, 5] = relative_position['angle']
            counter += 1

    return position_data


def transform_selected_lines_to_array(data, line_code_line_id_include_map):
    """The columns in the returned matrix are: lineCode, inputLine, otherLine, rowDiff, colDiff and angle."""
    total_data_counter = 0
    for line in data:
        if line['lineCode'] in line_code_line_id_include_map:
            lines_to_include = line_code_line_id_include_map[line['lineCode']]
            for relative_position in line['relativePositions']:
                # TODO This is temporary to see if the lines that are supposed to make out a rectangle stand out clearly in the clusters that are created
                if relative_position['inputLine'] not in lines_to_include or relative_position[
                    'otherLine'] not in lines_to_include:
                    continue

                total_data_counter += 1

    number_of_variables = 6
    position_data = np.zeros(total_data_counter * number_of_variables).reshape(total_data_counter, number_of_variables)
    counter = 0

    for line in data:
        line_code = line['lineCode']
        if line_code not in line_code_line_id_include_map:
            continue

        lines_to_include = line_code_line_id_include_map[line_code]
        if line['id'] not in lines_to_include:
            continue

        for relative_position in line['relativePositions']:

            # TODO This is temporary to see if the lines that are supposed to make out a rectangle stand out clearly in the clusters that are created
            if relative_position['inputLine'] not in lines_to_include or relative_position[
                'otherLine'] not in lines_to_include:
                continue

            position_data[counter, 0] = line_code
            position_data[counter, 1] = relative_position['inputLine']
            position_data[counter, 2] = relative_position['otherLine']
            position_data[counter, 3] = relative_position['rowDiff']
            position_data[counter, 4] = relative_position['colDiff']
            position_data[counter, 5] = relative_position['angle']
            counter += 1

    return position_data


def generate_line_data(data):
    result = {}
    for line in data:
        position_data = []
        for relative_position in line['relativePositions']:
            position_data.append(RelativePositionData(
                relative_position['rowDiff'],
                relative_position['colDiff'],
                relative_position['angle'],
                relative_position['otherLine']))

        result[line['id']] = LineData(line['id'], position_data)

    return result


def filter_out_four_last_lines_of_data(input_line_data):
    """ Returns a map between line codes (identifying the kanji) and a list with IDs of the four last lines. """
    line_code_line_ids_map = {}
    for line in input_line_data:
        if line['lineCode'] not in line_code_line_ids_map:
            line_code_line_ids_map[line['lineCode']] = {line['id']}
        else:
            line_code_line_ids_map[line['lineCode']].add(line['id'])

    result = {}
    for key, value in line_code_line_ids_map.items():
        line_ids = sorted(value)
        result[key] = line_ids[len(line_ids) - 4:]

    return result


def display_data(data):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(data[:, 0], data[:, 1], data[:, 2])
    plt.show()


if __name__ == '__main__':
    line_data = read_data()

    # data = transform_to_array(line_data)
    # line_data_map = generate_line_data(line_data)
