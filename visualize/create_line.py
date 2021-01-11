import numpy as np


def create_line(start_x: int, start_y: int, stop_x: int, stop_y: int):
    if start_x == stop_x:
        # Horizontal line
        rows = abs(stop_y - start_y) + 1
        result = np.empty((rows, 2), dtype=np.int32)

        if start_y < stop_y:
            result[:, 0] = np.repeat(start_x, rows)
            result[:, 1] = np.arange(start_y, stop_y + 1)
        else:
            result[:, 0] = np.repeat(start_x, rows)
            result[:, 1] = np.arange(stop_y, start_y + 1)

        return result

    if start_y == stop_y:
        # Vertical line
        rows = abs(stop_x - start_x) + 1
        result = np.full((rows, 2), np.NAN, dtype=np.int32)

        if start_x < stop_x:
            result[:, 0] = np.arange(start_x, stop_x + 1)
            result[:, 1] = np.repeat(start_y, rows)
        else:
            result[:, 0] = np.arange(stop_x, start_x + 1)
            result[:, 1] = np.repeat(start_y, rows)

        return result

    swap = stop_x < start_x
    first_translate = abs(min(0, min(start_x, stop_x)))
    second_translate = abs(min(0, min(start_y, stop_y)))

    if swap:
        start_x_translate = stop_x + first_translate
        start_y_translate = stop_y + second_translate
        stop_x_translate = start_x + first_translate
        stop_y_translate = start_y + second_translate
    else:
        start_x_translate = start_x + first_translate
        start_y_translate = start_y + second_translate
        stop_x_translate = stop_x + first_translate
        stop_y_translate = stop_y + second_translate

    x_delta = stop_x_translate - start_x_translate
    y_delta = stop_y_translate - start_y_translate
    delta_error = abs(y_delta / x_delta)

    if y_delta < 0:
        sign_y_delta = -1
    else:
        sign_y_delta = 1

    error = 0
    y = start_y_translate
    new_y = y

    array_shape = (2 * (abs(start_x - stop_x) + abs(start_y - stop_y)), 2)
    temp_result = np.full(array_shape, -1000, dtype=np.int32)
    counter = 0

    for x in np.arange(start_x_translate, stop_x_translate + 1):
        if y != new_y:
            if sign_y_delta < 0:
                for inc_y in np.arange(new_y, y + 1):
                    temp_result[counter, 0] = x
                    temp_result[counter, 1] = inc_y
                    counter = counter + 1
            else:
                for inc_y in np.arange(y, new_y + 1):
                    temp_result[counter, 0] = x
                    temp_result[counter, 1] = inc_y
                    counter = counter + 1
        else:
            temp_result[counter, 0] = x
            temp_result[counter, 1] = y
            counter = counter + 1

        y = new_y
        error = error + delta_error

        while error >= 0.5:
            new_y = new_y + sign_y_delta
            error = error - 1

    temp_result = temp_result[temp_result != -1000].reshape((-1, 2))

    temp_result[:, 0] = temp_result[:, 0] - first_translate
    temp_result[:, 1] = temp_result[:, 1] - second_translate

    if swap:
        temp_result[:, 0] = np.flip(temp_result[:, 0])
        temp_result[:, 1] = np.flip(temp_result[:, 1])

    return temp_result


def get_line_matrix(lines):
    x_min = 100000
    y_min = 100000

    x_max = -100000
    y_max = -100000

    for line in lines:
        (line_min_x, line_min_y) = np.amin(line, axis=0).astype(int)

        if x_min > line_min_x:
            x_min = line_min_x

        if y_min > line_min_y:
            y_min = line_min_y

        (line_max_x, line_max_y) = np.amax(line, axis=0).astype(int)

        if x_max < line_max_x:
            x_max = line_max_x

        if y_max < line_max_y:
            y_max = line_max_y

    print(x_min, x_max, y_min, y_max)
    image_matrix = np.full((x_max - x_min + 1, y_max - y_min + 1), 0, dtype=np.int32)

    x_offset = -x_min
    y_offset = -y_min

    line_value = 1
    for line in lines:
        for line_coordinate in line:
            # print(line_coordinate[0], line_coordinate[1])
            image_matrix[line_coordinate[0] + x_offset, line_coordinate[1] + y_offset] = line_value

        line_value += 1

    return image_matrix



if __name__ == '__main__':
    line_coordinates = create_line(10, 0, 5, 2)

    print(line_coordinates)
