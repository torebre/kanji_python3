from import_data import import_line_data
from visualize.create_line_svg import create_canvas, create_canvas2
import numpy as np
from import_data.import_data import filter_out_four_last_lines_of_data, transform_selected_lines_to_array, read_relation_data
from clustering.dbscan_clustering_test import extract_line_code_map_from_array, \
    extract_rectangle_relation_data_for_line_code, do_dbscan, add_label_data_to_line_code_map, \
    find_relation_sets_for_all_last_four_lines, create_cluster_colour_map, \
    create_line_colour_map_for_line_code

line_data = import_line_data.read_data()
line_relation_data = read_relation_data()

# The last four lines are the ones that make up a rectangle
last_four_lines = filter_out_four_last_lines_of_data(line_relation_data)

array_data = transform_selected_lines_to_array(line_relation_data, last_four_lines)
line_code_line_id_relation_data_map = extract_line_code_map_from_array(array_data)

data_used_for_clustering = array_data[:, 5]
dbscan_data = do_dbscan(np.reshape(data_used_for_clustering, (-1, 1)))
add_label_data_to_line_code_map(dbscan_data.labels_, array_data, line_code_line_id_relation_data_map)

line_code = 89
extract_rectangle_relation_data_for_line_code(line_code, line_code_line_id_relation_data_map, last_four_lines)

relation_sets = find_relation_sets_for_all_last_four_lines(last_four_lines, line_code_line_id_relation_data_map)
# for key in relation_sets:
#     print(key, ":")
#     for relation_data in relation_sets[key]:
#         print(relation_data)
#
#         line_line_colour_map = generate_color_map_for_line(relation_data, cluster_colour_map)
#
#         # print("Line colour map:", line_line_colour_map)
#
#         for first_line_id in line_line_colour_map:
#             line_colours = line_line_colour_map[first_line_id]
#             line_to_line_colour_map[(key, first_line_id)] = line_colours
#
# print("Line to line colour map: ", line_to_line_colour_map)

cluster_colours = create_cluster_colour_map(set(dbscan_data.labels_))
colour_map = create_line_colour_map_for_line_code(line_code, relation_sets, cluster_colours)
transformed_colour_map = {}

for key in colour_map:
    colour_code = colour_map[key]

    if int(key[0]) in transformed_colour_map:
        transformed_colour_map[int(key[0])].append(colour_code)
    else:
        transformed_colour_map[int(key[0])] = [colour_code]

    if int(key[1]) in transformed_colour_map:
        transformed_colour_map[int(key[1])].append(colour_code)
    else:
        transformed_colour_map[int(key[1])] = [colour_code]


is_line_1 = line_data['unicode'] == line_code
line_data_1 = line_data[is_line_1]

# canvas = create_canvas(line_data_1)

# colour_map = {1.0: ["green", "yellow", "red"]}
canvas = create_canvas2(line_data_1, transformed_colour_map)
canvas.setPixelScale(5)
canvas.savePng('test_output_svg2.png')
