import math

import numpy as np
import numpy.typing as npt
import pandas as pd

from line_data_generation.generate_training_sample import generate_training_samples


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


if __name__ == "__main__":
    training_samples = generate_training_samples()
    test_sample = training_samples[0]
    number_of_lookup_examples = len(training_samples) - 1

    # There are the same number of lines in all the samples
    number_of_lines_in_examples = len(training_samples[0])

    # dataframe = pd.DataFrame(data=np.zeros(shape=(8, number_of_lookup_examples * (number_of_lookup_examples - 1))),
    # columns=["unicode", "line_number", "angle", "length", "start_x", "start_y", "index_line1",
    #          "index_line2"])

    data = np.zeros((number_of_lookup_examples * number_of_lines_in_examples * (number_of_lines_in_examples - 1), 5))

    example_counter = 0
    for sample in training_samples[1:]:
        count = 1

        for line in sample:
            count2 = 1

            for line2 in sample:
                if count == count2:
                    count2 += 1
                    continue

                (angle_diff, midpoint_x_diff, midpoint_y_diff) = describe_two_lines(line, line2)
                data[example_counter] = [angle_diff, midpoint_x_diff, midpoint_y_diff, count, count2]
                example_counter += 1
                count2 += 1

                # print(f"{angle_diff}, {midpoint_x_diff}, {midpoint_y_diff}")

            count += 1

    print(example_counter)

    # print(data)
