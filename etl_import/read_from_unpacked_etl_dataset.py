import os
from typing import List

import matplotlib.pyplot as plt
import numpy.typing as npt

from matplotlib import pyplot
import imageio

# This is the directory where the data has been stored
# after the data extraction has been run
from image_data_handling.clean_etl_images import clean_image

base_directory = '/home/student/projects/etlcdb-image-extractor/output/'


def list_dataset_directory_contents(base_path: str):
    for file in os.listdir(base_path):
        print(file)


def read_samples_for_each_kanji_in_dataset(dataset_part_path: str) -> List[npt.ArrayLike]:
    images: List[npt.ArrayLike] = []

    for file in os.listdir(dataset_part_path):
        # The name of the directories are the unicodes, in hex, for
        # the kanji examples contained in that directory
        full_path = os.path.join(dataset_part_path, file)
        if os.path.isdir(full_path):
            # A directory containing png-images of kanji
            images.extend(read_samples_from_folder(full_path))

    return images


def read_samples_from_folder(sample_directory: str, number_of_samples_to_read: int) -> List[npt.ArrayLike]:
    images = []
    counter = 0

    for written_kanji_file in os.listdir(sample_directory):
        if written_kanji_file == ".char.txt":
            # This is a file containing just the character
            # the samples in this directory are for
            continue

        kanji_full_path = os.path.join(sample_directory, written_kanji_file)
        image = imageio.imread(kanji_full_path)

        images.append(image)

        counter += 1
        if counter >= number_of_samples_to_read:
            break

    return images


if __name__ == '__main__':
    # list_dataset_directory_contents(base_directory)
    # read_samples_for_each_kanji_in_dataset('/home/student/projects/etlcdb-image-extractor/output/ETL9G')
    test_images = read_samples_from_folder('/home/student/projects/etlcdb-image-extractor/output/ETL9G/0x8c9d', 3)

    for image in test_images:
        # plt.imshow(image)
        # plt.show()

        cleaned_image = clean_image(image)
        plt.imshow(cleaned_image)
        plt.show()
