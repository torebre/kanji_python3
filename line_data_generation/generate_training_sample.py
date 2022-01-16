import random
import math

import numpy as np
import numpy.typing as npt

import pandas as pd

from visualize.create_line_svg import draw_line_data_on_svg_canvas


def generate_training_sample(number_of_rows: int = 64, number_of_columns: int = 64,
                             number_of_random_lines: int = 10, include_rectangle: bool = True) -> npt.ArrayLike:
    """
    Generates training samples which are a collection of lines.
    If include_rectangle the last four lines represent a rectangle.

    :param number_of_rows:
    :param number_of_columns:
    :param number_of_random_lines:
    :param include_rectangle:
    :return: A multidimensional array where the number of rows is the number of lines, and the columns
    are these: Angle, line length, start x coordinate, start y coordinate, stop x coordinate, stop y coordinate
    """
    training_samples = np.zeros((number_of_random_lines + 4, 6))

    if not include_rectangle:
        # Include four more random lines so that the returned number of lines is always the same
        number_of_random_lines += 4

    for i in range(0, number_of_random_lines):
        # TODO Check if end is inclusive for range
        start_x = random.sample(range(0, number_of_rows), 1)[0]
        start_y = random.sample(range(0, number_of_columns), 1)[0]

        stop_x = random.sample(range(0, number_of_rows), 1)[0]
        stop_y = random.sample(range(0, number_of_columns), 1)[0]

        x_delta = stop_x - start_x
        y_delta = stop_y - start_y

        # TODO Check that this is the correct function to use
        angle = math.atan2(y_delta, x_delta)
        line_length = math.sqrt(x_delta * x_delta + y_delta * y_delta)

        training_samples[i, 0] = angle
        training_samples[i, 1] = line_length
        training_samples[i, 2] = start_x
        training_samples[i, 3] = start_y
        training_samples[i, 4] = stop_x
        training_samples[i, 5] = stop_y

    if include_rectangle:
        training_samples[number_of_random_lines:(number_of_random_lines + 4), :] = add_rectangle(number_of_rows,
                                                                                                  number_of_columns)

    return training_samples


def add_rectangle(number_of_rows: int = 64, number_of_columns: int = 64) -> npt.ArrayLike:
    start_x = random.sample(range(0, number_of_rows - 1), 1)[0]
    available_space = number_of_columns - start_x
    line_length = random.sample(range(1, available_space), 1)[0]

    start_y = random.sample(range(0, number_of_columns - 1), 1)[0]
    available_space_y = number_of_rows - start_y
    line_length_y = random.sample(range(1, available_space_y), 1)[0]

    return np.array([[0, line_length, start_x, start_y, start_x + line_length, start_y],
                     [0.5 * math.pi, line_length_y, start_x, start_y, start_x, start_y + line_length_y],
                     [0, line_length, start_x, start_y + line_length_y, start_x + line_length, start_y + line_length_y],
                     [(3 / 2) * math.pi, line_length_y, start_x + line_length, start_y + line_length_y,
                      start_x + line_length,
                      start_y]])


def generate_training_samples(total_number_of_samples: int = 100,
                              number_of_samples_to_not_include_rectangles: int = 0,
                              number_of_random_lines: int = 10) -> npt.ArrayLike:
    """
    Any samples with rectangles are included first

    :param total_number_of_samples:
    :param number_of_samples_to_not_include_rectangles:
    :param number_of_random_lines:
    :return:
    """
    samples = []
    for sample_counter in range(total_number_of_samples):
        samples.append(generate_training_sample(64, 64, number_of_random_lines, sample_counter <= total_number_of_samples - number_of_samples_to_not_include_rectangles))

    return samples


if __name__ == '__main__':
    training_samples = generate_training_samples()

    rows_in_sample = len(training_samples[0])

    print("Rows in sample: ", rows_in_sample)

    transformed_sample = np.zeros((rows_in_sample, 6))
    for i in range(rows_in_sample):
        transformed_sample[i, 0] = 1
        transformed_sample[i, 1] = i
        transformed_sample[i, 2:6] = training_samples[0][i, 0:4]

    dataframe = pd.DataFrame(transformed_sample,
                             columns=["unicode", "line_number", "angle", "length", "start_x", "start_y"])

    canvas = draw_line_data_on_svg_canvas(dataframe)
    canvas.setPixelScale(5)
    canvas.savePng('test_output_svg2.png')
