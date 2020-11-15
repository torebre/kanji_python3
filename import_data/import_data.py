import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits import mplot3d

# from typing import TypedDict

from import_data.RelativePositionData import RelativePositionData
from import_data.LineData import LineData

from import_data.RelationDataExtraction import RelationDataExtraction


def read_data():
    with open('/home/student/workspace/testEncodings/line_relative_position_information_rectangle.json') as input_data:
        data = json.load(input_data)
        return data


def transform_to_array(data):
    total_data_counter = 0
    for line in data:
        for _ in line['relativePositions']:
            total_data_counter += len(line['relativePositions'])

    # position_data = np.ndarray(np.shape(total_data_counter, 3))
    position_data = np.zeros(total_data_counter * 3).reshape(total_data_counter, 3)
    counter = 0

    for line in data:
        for relative_position in line['relativePositions']:
            position_data[counter, 0] = relative_position['rowDiff']
            position_data[counter, 1] = relative_position['colDiff']
            position_data[counter, 2] = relative_position['angle']
            counter += 1

    return position_data

# class LinePositionMap(TypedDict):
#     line_id: int
#     line_data: LineData


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


def create_cluster(data):
    kmeans = KMeans(n_clusters=4)

    kmeans.fit(data)
    print(kmeans.cluster_centers_)

    return kmeans


def diplay_data(data):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(data[:, 0], data[:, 1], data[:, 2])
    plt.show()


if __name__ == '__main__':
    line_data = read_data()
    # data = transform_to_array(line_data)
    # fitted_kmeans = create_cluster(data)
    # diplay_data(data)

    line_data_map = generate_line_data(line_data)
    # print("Line data:", line_data_map[1])

    relation_data_extraction = RelationDataExtraction()
    relation_data_extraction.extract_relation_data_for_all_lines(line_data_map)
