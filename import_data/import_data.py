import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def read_data():
    with open('/home/student/workspace/testEncodings/line_relative_position_information_rectangle.json') as input_data:
        data = json.load(input_data)
        return data


def transform_to_array(data):
    total_data_counter = 0
    for line in data:  # ['relativePositions']:
        for _ in line['relativePositions']:
            total_data_counter += len(line['relativePositions'])

    # position_data = np.ndarray(np.shape(total_data_counter, 3))
    position_data = np.zeros(total_data_counter * 3).reshape(total_data_counter, 3)
    counter = 0

    for line in data:  # ['relativePositions']:
        for relative_position in line['relativePositions']:
            position_data[counter, 0] = relative_position['rowDiff']
            position_data[counter, 1] = relative_position['colDiff']
            position_data[counter, 2] = relative_position['angle']
            counter += 1

    return position_data


def create_cluster(data):
    kmeans = KMeans(n_clusters=4)

    kmeans.fit(data)
    print(kmeans.cluster_centers_)


if __name__ == '__main__':
    line_data = read_data()
    data = transform_to_array(line_data)
    create_cluster(data)
