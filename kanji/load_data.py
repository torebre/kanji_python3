import os

import matplotlib.pyplot as plt
import numpy as np




def loadFilesInDirectory(path):
    for file in os.listdir(path):
        file_with_path = os.path.join(path, file)
        if os.path.isfile(file_with_path):
            yield np.loadtxt(file_with_path, delimiter=",")


def plotImages(images):
    for image in images:
        plt.imshow(image) #, cmap='hot', interpolation='nearest')
        plt.show()

