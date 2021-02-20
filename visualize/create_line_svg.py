import drawSvg as draw
import pandas as pd
import numpy as np
from drawSvg import Drawing

from import_data import import_line_data
from visualize.DrawLines import generate_line_coordinates
from visualize.create_line import get_borders


def draw_line(start_x: int, start_y: int, stop_x: int, stop_y: int, canvas, stroke_colour='white', stroke_width=2):
    canvas.append(draw.Line(start_x, start_y, stop_x, stop_y, stroke=stroke_colour, stroke_width=stroke_width))


def create_canvas(line_data: pd.DataFrame, colour_map={}):
    line_coordinates = generate_line_coordinates(line_data)
    (x_min, y_min, x_max, y_max) = get_borders(line_coordinates)

    drawing = draw.Drawing(x_max - x_min, y_max - y_min)

    for index, line in line_data.iterrows():
        angle = line['angle']
        line_length = line['length']
        start_x = line['start_x'].astype(int)
        start_y = line['start_y'].astype(int)

        stop_x = start_x + np.rint(line_length * np.cos(angle)).astype(int)
        stop_y = start_y + np.rint(line_length * np.sin(angle)).astype(int)

        print("Line number: ", line['line_number'])

        if line['line_number'] in colour_map:
            draw_line(start_x, start_y, stop_x, stop_y, drawing, colour_map[line['line_number']])
        else:
            draw_line(start_x, start_y, stop_x, stop_y, drawing)

    return drawing


def create_canvas2(line_data: pd.DataFrame, colour_map={}):
    line_coordinates = generate_line_coordinates(line_data)
    (x_min, y_min, x_max, y_max) = get_borders(line_coordinates)

    drawing = draw.Drawing(x_max - x_min, y_max - y_min)
    for index, line in line_data.iterrows():
        angle = line['angle']
        line_length = line['length']
        start_x = line['start_x'].astype(int)
        start_y = line['start_y'].astype(int)

        stop_x = start_x + np.rint(line_length * np.cos(angle)).astype(int)
        stop_y = start_y + np.rint(line_length * np.sin(angle)).astype(int)

        print("Line number: ", line['line_number'])

        if line['line_number'] in colour_map:
            draw_line_with_colours(start_x, start_y, stop_x, stop_y, drawing, colour_map[line['line_number']])
        else:
            draw_line(start_x, start_y, stop_x, stop_y, drawing)

    return drawing


def draw_line_with_colours(start_x, start_y, stop_x, stop_y, drawing: Drawing, colours):
    number_of_colours = len(colours)

    initial_offset = number_of_colours / 2
    counter = 0

    for colour in colours:
        x_diff = stop_x - start_x
        y_diff = stop_y - start_y

        offset = initial_offset - counter
        counter += 1

        if x_diff == 0:
            drawing.append(draw.Line(start_x + offset, start_y,
                                     stop_x + offset, stop_y,
                                     stroke=colour, stroke_width=1))
        else:
            y2 = 1
            y1 = -(y_diff/x_diff)

            length = np.sqrt(y1 * y1 + y2)
            y1_norm = y1 / length
            y2_norm = y2 / length

            drawing.append(draw.Line(start_x + y1_norm * offset, start_y + y2_norm * offset,
                                stop_x + y1_norm * offset, stop_y + y2_norm * offset,
                                stroke=colour, stroke_width=1))


if __name__ == '__main__':
    line_data = import_line_data.read_data()

    is_line_1 = line_data['unicode'] == 86
    line_data_1 = line_data[is_line_1]

    colour_map = {1.0: ["#1248ff", "#fefefe", "red"]}
    canvas = create_canvas2(line_data_1, colour_map)
    canvas.setPixelScale(5)
    canvas.savePng('test_output_svg.png')

