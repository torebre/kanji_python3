import tensorflow as tf
import struct
import numpy as np
from matplotlib import pylab as plt

from skimage.morphology import skeletonize
from skimage.transform import resize

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/")




structFormat = 'hh4s504s64s'
structLength = struct.calcsize(structFormat)
unpackFunction = struct.Struct(structFormat).unpack_from

kanjiData = []

inputImagesTemp = []
inputClasses = []

numberOfTrainingSamples = 5

with open('/home/student/Downloads/ETL/ETL9B/ETL9B_2', mode = 'rb') as file:
    for i in range(1, numberOfTrainingSamples):
        record = file.read(structLength)

        (serialSheetNumber, kanjiCode, typicalReading, imageData, uncertain) = unpackFunction(record)
        image = np.unpackbits(np.fromstring(imageData, dtype=np.uint8)).reshape((63, 64))

        skeletonizedImage = skeletonize(image)
        resizedImage = resize(skeletonizedImage, (28, 28))

        kanjiData.append(tuple([serialSheetNumber, kanjiCode, typicalReading, resizedImage]))

        inputImagesTemp.append(resizedImage)
        inputClasses.append(kanjiCode)


tf.reset_default_graph()

np.random.seed(10)
tf.set_random_seed(10)

inputImages = np.asarray(inputImagesTemp)



X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")

caps1NumberOfMaps = 32
caps1NumberOfCapsules = caps1NumberOfMaps * 6 * 6
caps1NumberOfDimensions = 8


conv1_params = {
    "filters": 256,
    "kernel_size": 9,
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu
}

conv2_params = {
    "filters": caps1NumberOfMaps * caps1NumberOfDimensions,
    "kernel_size": 9,
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu
}


conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)

caps1Raw = tf.reshape(conv2, [-1, caps1NumberOfCapsules, caps1NumberOfDimensions], name="caps1_raw")


def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector


caps1Output = squash(caps1Raw, name="caps1_output")

numberOfOutputs = 3036

# One output for each of the different kanjis
caps2NumberOfCapsules = 3036
caps2NumberOfDimensions = 16

initSigma = 0.1

weightsInit = tf.random_normal(shape=(1, caps1NumberOfCapsules, caps2NumberOfCapsules, caps2NumberOfDimensions, caps1NumberOfDimensions))



weights = tf.Variable(weightsInit, name="W")
batchSize = tf.shape(X)[0]

weightsTiled = tf.tile(weights, [batchSize, 1, 1, 1, 1], name="W_tiled")

caps1OutputExpanded = tf.expand_dims(caps1Output, -1, name="caps1_output_expanded")
caps1OutputTile = tf.expand_dims(caps1OutputExpanded, 2, name="caps1_output_tile")
caps1OutputTiled = tf.tile(caps1OutputTile, [1, 1, caps2NumberOfCapsules, 1, 1], name="caps1_output_tiled")

caps2Predicted = tf.matmul(weightsTiled, caps1OutputTiled, name="caps2_predicted")

rawWeights = tf.zeros([batchSize, caps1NumberOfCapsules, caps2NumberOfCapsules, 1, 1],
                      dtype=np.float32, name="raw_weights")


routingWeights = tf.nn.softmax(rawWeights, dim=2, name="routing_weights")

weightedPredictions = tf.multiply(routingWeights, caps2Predicted, name="weighted_predictions")
weightedSum = tf.reduce_sum(weightedPredictions, axis=1, keep_dims=True, name="weighted_sum")

caps2OutputRound1 = squash(weightedSum, axis=-2, name="caps2_output_round_1")

caps2OutputRound1Tiled = tf.tile(caps2OutputRound1, [1, caps1NumberOfCapsules, 1, 1, 1], name="caps2_output_round_1_tiled")
agreement = tf.matmul(caps2Predicted, caps2OutputRound1Tiled, transpose_a=True, name="agreement")

rawWeightsRound2 = tf.add(rawWeights, agreement, name="raw_weights_round_2")

routingWeightsRound2 = tf.nn.softmax(rawWeightsRound2, dim=2, name="routing_weights_round_2")
weightedPredictionsRound2 = tf.multiply(routingWeightsRound2, caps2Predicted, name="weighted_predictions_round_2")
weightedSumRound2 = tf.reduce_sum(weightedPredictionsRound2, axis=1, keep_dims=True, name="weighted_sum_round_2")
caps2OutputRound2 = squash(weightedSumRound2, axis=-2, name="caps2_output_round_2")

caps2Output = caps2OutputRound2

def safeNorm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)

yProbability = safeNorm(caps2Output, axis=-2, name="y_proba")
yProbabilityArgmax = tf.argmax(yProbability, axis=2, name="y_proba")

yPrediction = tf.squeeze(yProbabilityArgmax, axis=[1, 2], name="y_pred")

# Labels
y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

mPlus = 0.9
mMinus = 0.1
lambdaParameter = 0.5

T = tf.one_hot(y, depth=caps2NumberOfCapsules, name="T")

caps2OutputNorm = safeNorm(caps2Output, axis=-2, keep_dims=True, name="caps2_output_norm")

presentErrorRaw = tf.square(tf.maximum(0., mPlus - caps2OutputNorm), name="present_error_raw")
presentError = tf.reshape(presentErrorRaw, shape=(-1, numberOfOutputs), name="present_error")

absentErrorRaw = tf.square(tf.maximum(0., caps2OutputNorm - mMinus), name="absent_error_raw")
absentError = tf.reshape(absentErrorRaw, shape=(-1, numberOfOutputs), name="absent_error")

L = tf.add(T * presentError, lambdaParameter * (1.0 - T) * absentError, name="L")

marginLoss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")


# Mask
maskWithLabels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")

reconstructionTargets = tf.cond(maskWithLabels, lambda: y, lambda: yPrediction, name="reconstruction_targets")

reconstructionMask = tf.one_hot(reconstructionTargets, depth=caps2NumberOfCapsules, name="reconstruction_mask")

reconstructionMaskReshaped = tf.reshape(reconstructionMask, [-1, 1, caps2NumberOfCapsules, 1, 1], name="reconstruction_mask_reshaped")

caps2OutputMasked = tf.multiply(caps2Output, reconstructionMaskReshaped, name="caps2_output_masked")

decoderInput = tf.reshape(caps2OutputMasked, [-1, caps2NumberOfCapsules * caps2NumberOfDimensions], name="decoder_input")


# Decoder
nHidden1 = 512
nHidden2 = 1024
nOutput = 28 * 28

with tf.name_scope("decoder"):
    hidden1 = tf.layers.dense(decoderInput, nHidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, nHidden2, activation=tf.nn.relu, name="hidden2")
    decoderOutput = tf.layers.dense(hidden2, nOutput, activation=tf.nn.sigmoid, name="decoder_output")


# Reconstruction loss
xFlat = tf.reshape(X, [-1, nOutput], name="X_flat")
squaredDifference = tf.square(xFlat - decoderOutput, name="squared_difference")
reconstructionLoss = tf.reduce_mean(squaredDifference, name="reconstruction_loss")

# Final loss
alpha = 0.0005
loss = tf.add(marginLoss, alpha * reconstructionLoss, name="loss")

# Accuracy
correct = tf.equal(y, yPrediction, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32, name="accuracy"))

# Training operations
optimizer = tf.train.AdamOptimizer()
trainingOptimizer = optimizer.minimize(loss, name="training_op")

# Init and saver
init = tf.global_variables_initializer()
saver = tf.train.Saver()


# Training
numberOfEpochs = 10
batchSize = 50
restoreCheckpoint = True

numberOfIterationsPerEpoch = numberOfTrainingSamples # // batchSize

# TODO This is just to test that things work, should use a different set for validation
numberOfIterationsValidation = numberOfTrainingSamples // batchSize
bestLossVal = np.infty
checkpointPath = "./etl_capsule_network"


with tf.Session() as sess:
    if restoreCheckpoint and tf.train.checkpoint_exists(checkpointPath):
        saver.restore(sess, checkpointPath)
    else:
        init.run()

    print "Test24: ", numberOfEpochs

    for epoch in range(numberOfEpochs):
        startOfNextMiniBatch = 0

        print "Test25: ", numberOfIterationsPerEpoch

        for iteration in range(1, numberOfIterationsPerEpoch + 1):
            xBatch = inputImages[startOfNextMiniBatch:(startOfNextMiniBatch + batchSize)]
            yBatch = inputClasses[startOfNextMiniBatch:(startOfNextMiniBatch + batchSize)]

            startOfNextMiniBatch = np.mod(startOfNextMiniBatch + batchSize, numberOfTrainingSamples)

            print "Test23"

            _, lossTraining = sess.run([trainingOptimizer, loss],
                                       feed_dict={X: xBatch.reshape([-1, 28, 28, 1]),
                                                  y: yBatch,
                                                  maskWithLabels: True})

            print("\rIteration: {}/{} ({:.1f}%) Loss: {:.5f}".format(iteration,
                                                                     numberOfIterationsPerEpoch,
                                                                     iteration * 100 / numberOfIterationsPerEpoch,
                                                                     lossTraining))

        lossVals = []
        accVals = []
        startOfNextMiniBatch = 0
        for iteration in range(1, numberOfIterationsValidation + 1):
            xBatch = inputImages[startOfNextMiniBatch:(startOfNextMiniBatch + batchSize)]
            yBatch = inputClasses[startOfNextMiniBatch:(startOfNextMiniBatch + batchSize)]

            startOfNextMiniBatch = np.mod(startOfNextMiniBatch + batchSize, numberOfTrainingSamples)

            lossVal, accVal = sess.run([loss, accuracy],
                                           feed_dict={X: xBatch.reshape([-1, 28, 28, 1]),
                                                      y: yBatch})

            lossVals.append(lossVal)
            accVals.append(accVal)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(iteration, numberOfIterationsValidation,
                                                                       iteration * 100 / numberOfIterationsValidation))
            lossVal = np.mean(lossVals)
            accVal = np.mean(accVals)
            print("\rEpoch: {} Val accuracy: {:.4f}% Loss: {:.6f}{}".format(epoch + 1, accVal * 100, lossVal,
                                                                                " (improved)" if lossVal < bestLossVal else ""))

            if lossVal < bestLossVal:
                savePath = saver.save(sess, checkpointPath)
                bestLossVal = lossVal