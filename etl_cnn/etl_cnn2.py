import tensorflow as tf
import struct
import numpy as np

from skimage.morphology import skeletonize
from skimage.transform import resize

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras import backend as K

from tensorflow.python.framework import graph_util




structFormat = 'hh4s504s64s'
structLength = struct.calcsize(structFormat)
unpackFunction = struct.Struct(structFormat).unpack_from

kanjiData = []

inputImagesTemp = []
inputClasses = []

numberOfTrainingSamples = 100000

classesMap = {}
counter = 0

with open('/home/student/Downloads/ETL/ETL9B/ETL9B_2', mode = 'rb') as file:
    # The first record is a dummy record
    record = file.read(structLength)

    for i in range(0, numberOfTrainingSamples):
        record = file.read(structLength)

        (serialSheetNumber, kanjiCode, typicalReading, imageData, uncertain) = unpackFunction(record)
        image = np.unpackbits(np.fromstring(imageData, dtype=np.uint8)).reshape((63, 64))

        skeletonizedImage = skeletonize(image)
        resizedImage = np.asarray(resize(skeletonizedImage, (28, 28)) > 0, dtype=int)

        kanjiData.append(tuple([serialSheetNumber, kanjiCode, typicalReading, resizedImage]))

        inputImagesTemp.append(resizedImage)

        if kanjiCode not in classesMap:
            classesMap[kanjiCode] = counter
            counter = counter + 1

        inputClasses.append(classesMap[kanjiCode])





xTrain = np.asarray(inputImagesTemp[0:90000]).reshape(90000, 28, 28, 1)
xTest = np.asarray(inputImagesTemp[90000:100000]).reshape(10000, 28, 28, 1)

allClasses = to_categorical(inputClasses)
yTrain = allClasses[0:90000]
yTest = allClasses[90000:100000]

model = Sequential()

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(counter, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=1)

predictFirst = model.predict(xTest[:1])
np.argmax(predictFirst)




output_node_names = 'dense_1/Softmax'
output_graph_def = graph_util.convert_variables_to_constants(
    K.get_session(),
    tf.get_default_graph().as_graph_def(),
    output_node_names.split(",")
)
model_file = "./saved_model.pb"
with tf.gfile.GFile(model_file, "wb") as f:
    f.write(output_graph_def.SerializeToString())



print([k for k,v in classesMap.items() if v == 87])