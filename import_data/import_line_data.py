import pandas as pandas


# def read_data():
#     with open(
#             '/home/student/workspace/kanjiR/training_data.csv') as input_data:
#         data = json.load(input_data)
#         return data


def read_data() -> pandas.DataFrame:
    return pandas.read_csv('/home/student/workspace/kanjiR/training_data.csv')


if __name__ == '__main__':
    line_data = read_data()

    # print(line_data.head(10))

    is_line_1 = line_data['unicode'] == 1
    line_data_1 = line_data[is_line_1]

    print(line_data_1)
