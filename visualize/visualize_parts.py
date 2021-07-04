from import_data import import_line_data
from visualize.create_line_svg import draw_line_data_on_svg_canvas

if __name__ == '__main__':
    line_data = import_line_data.read_data()

    is_line_1 = line_data['unicode'] == 86
    line_data_1 = line_data[is_line_1]

    colour_map = {1.0: ["#1248ff", "#fefefe", "red"]}
    canvas = draw_line_data_on_svg_canvas(line_data_1, colour_map)
    canvas.setPixelScale(5)
    canvas.savePng('test_output_svg.png')

