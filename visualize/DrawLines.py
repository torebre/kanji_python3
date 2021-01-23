import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualize.create_line import create_line
from visualize.create_line import get_line_matrix
from import_data import import_line_data


def generate_line_coordinates(line_data_frame: pd.DataFrame):
    lines = []

    for index, line in line_data_frame.iterrows():
        angle = line['angle']
        line_length = line['length']
        start_x = line['start_x'].astype(int)
        start_y = line['start_y'].astype(int)

        stop_x = start_x + np.rint(line_length * np.cos(angle)).astype(int)
        stop_y = start_y + np.rint(line_length * np.sin(angle)).astype(int)

        line_points = create_line(start_x, start_y, stop_x, stop_y)
        lines.append(line_points)

    return lines


if __name__ == '__main__':
    line_data = import_line_data.read_data()

    is_line_1 = line_data['unicode'] == 1
    line_data_1 = line_data[is_line_1]

    line_coordinates = generate_line_coordinates(line_data_1)
    line_matrix = get_line_matrix(line_coordinates)

    # fig = plt.figure()

    plt.matshow(line_matrix)
    plt.show()