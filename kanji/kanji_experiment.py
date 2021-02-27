import numpy as np
import tensorflow as tf
from rbm.showRbm import showWeights
from rbm.cd1 import cd1

import kanji.load_data as ld




learningRate = 0.9
momentumSpeedConstant = 0.9
miniBatchSize = 50
numberOfHiddenUnits = 30
numberOfIterations = 1000
numberType = "float32"




images = ld.loadFilesInDirectory("/home/student/workspace/testEncodings/fragments2")

# ld.plotImages(images)

loadedImages = []
imageShape = None

for image in images:
    loadedImages.append(np.reshape(image.astype("float32"), (image.shape[0] * image.shape[1])))

    if imageShape is None:
        imageShape = image.shape

inputs = np.asarray(loadedImages).transpose()

initialWeights = ((np.random.random_sample((numberOfHiddenUnits, inputs.shape[0])) * 2 - 1) * 0.1).astype(numberType)
weights = tf.Variable(initial_value=initialWeights, name="weights")

writer = tf.summary.FileWriter("../model_output/kanji_logs")

momentumSpeed = tf.Variable(np.zeros(weights.shape).astype(numberType), dtype=numberType)

weightSummary = tf.summary.image('input', tf.reshape(weights[0, :], [-1, imageShape[0], imageShape[1], 1]), 1)

startOfNextMiniBatch = 0

miniBatch = tf.placeholder(dtype=numberType, shape=[inputs.shape[0], miniBatchSize], name="miniBatch")
gradient = cd1(miniBatch, weights)

momentumSpeedUpdated = tf.add(tf.multiply(tf.constant(momentumSpeedConstant, dtype=numberType), momentumSpeed), gradient)
weightsUpdated = tf.add(weights, tf.multiply(momentumSpeedUpdated, tf.constant(learningRate, dtype=numberType)))

updateMomentumSpeed = momentumSpeed.assign(momentumSpeedUpdated)
updateWeights = weights.assign(weightsUpdated)

for i in range(numberOfHiddenUnits):
    tf.summary.image('weights' + str(i), tf.reshape(updateWeights[i, :], [-1, 32, 32, 1]), 1)


tf.summary.histogram('momentumSpeedUpdated', momentumSpeedUpdated[0])
tf.summary.histogram('weightsUpdated',weightsUpdated[0])

merged_summary_op = tf.summary.merge_all()

number_of_batches = inputs.shape[1] / miniBatchSize

with tf.Session() as sess:
    momentumSpeed.initializer.run()
    weights.initializer.run()

    for i in range(numberOfIterations):
        startOfNextMiniBatch = 0
        for j in range(number_of_batches):
            miniBatchValues = inputs[:, startOfNextMiniBatch: (startOfNextMiniBatch + miniBatchSize)]
            startOfNextMiniBatch = np.mod(startOfNextMiniBatch + miniBatchSize, inputs.shape[1])

            _, _, summary = sess.run([updateWeights, updateMomentumSpeed, merged_summary_op], feed_dict={miniBatch: miniBatchValues})

        print("Iteration: ", i)

        if i % 1000 == 0:
            writer.add_summary(summary, i)

    finalWeights = sess.run(weights)
    showWeights(finalWeights, imageShape)
