import drawSvg as draw
import pandas as pd
import numpy as np

from import_data import import_line_data
from visualize.DrawLines import generate_line_coordinates
from visualize.create_line import get_borders


def draw_line(start_x: int, start_y: int, stop_x: int, stop_y: int, canvas):
    canvas.append(draw.Line(start_x, start_y, stop_x, stop_y, stroke='white'))


def create_canvas(line_data: pd.DataFrame):
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

        draw_line(start_x, start_y, stop_x, stop_y, drawing)

    return drawing



if __name__ == '__main__':
    line_data = import_line_data.read_data()

    is_line_1 = line_data['unicode'] == 1
    line_data_1 = line_data[is_line_1]

    canvas = create_canvas(line_data_1)
    canvas.setPixelScale(5)
    canvas.savePng('test_output_svg.png')

