import numpy as np


def create_line(start_x: int, start_y: int, stop_x: int, stop_y: int):
    if start_x == stop_x:
        # Horizontal line
        rows = abs(stop_y - start_y) + 1
        result = np.empty((rows, 2))

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
        result = np.full((rows, 2), np.NAN)

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

    array_shape = (2 * abs(start_x - stop_x) + abs(start_y - stop_y), 2)
    temp_result = np.full(array_shape, np.NAN)
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

    temp_result = temp_result[~np.isnan(temp_result)].reshape((-1, 2))

    temp_result[:, 0] = temp_result[:, 0] - first_translate
    temp_result[:, 1] = temp_result[:, 1] - second_translate

    if swap:
        temp_result[:, 0] = np.flip(temp_result[:, 0])
        temp_result[:, 1] = np.flip(temp_result[:, 1])

    return temp_result


if __name__ == '__main__':
    line_coordinates = create_line(10, 0, 5, 2)

    print(line_coordinates)
