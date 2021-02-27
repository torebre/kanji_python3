import scipy.io as sio

import numpy as np
import tensorflow as tf

from showRbm import showWeights

dataSet = sio.loadmat("/home/student/Documents/Neural networks for Machine Learning/assignment4/data_set.mat")

learningRate = 0.9
momentumSpeedConstant = 0.9
miniBatchSize = 100
numberOfHiddenUnits = 50
numberOfIterations = 100000
numberType = "float32"

inputs = dataSet["data"]["training"][0, 0]["inputs"][0, 0].astype(numberType)

initialWeights = ((np.random.random_sample((numberOfHiddenUnits, inputs.shape[0])) * 2 - 1) * 0.1).astype(numberType)
weights = tf.Variable(initial_value=initialWeights, name="weights")

writer = tf.summary.FileWriter("../model_output/logs35")

momentumSpeed = tf.Variable(np.zeros(weights.shape).astype(numberType), dtype=numberType)
startOfNextMiniBatch = 0

miniBatch = tf.placeholder(dtype=numberType, shape=[inputs.shape[0], miniBatchSize], name="miniBatch")


# Start implementation of contrastive divergence with 1 step (CD1)
visible_data_sampled = tf.cast(tf.greater_equal(miniBatch, tf.random_uniform(miniBatch.shape)), numberType)

# Visible state to hidden probabilities
ones = tf.constant(1, numberType, [weights.shape[0], visible_data_sampled.shape[1]])
multiply_term = tf.exp(tf.matmul(tf.negative(weights), visible_data_sampled))

hidden_probabilities = tf.div(ones, tf.add(ones, multiply_term))
hidden_states = tf.cast(tf.greater_equal(hidden_probabilities, tf.random_uniform(hidden_probabilities.shape)), numberType)

# Compute the hidden state to visible probabilities
ones_hidden = tf.constant(1, numberType, miniBatch.shape)
multiply_term_hidden = tf.exp(tf.matmul(tf.transpose(tf.negative(weights)), hidden_states))
visible_probabilities = tf.div(ones_hidden, tf.add(ones_hidden, multiply_term_hidden))

visible_states = tf.cast(tf.greater_equal(visible_probabilities, tf.random_uniform(miniBatch.shape)), numberType)

hidden_probabilities2 = tf.div(ones, tf.add(ones, tf.exp(tf.matmul(tf.negative(weights), visible_states))))

numberOfConfigurations = tf.constant(miniBatch.shape[1].value, dtype = numberType)

term1 = tf.div(tf.matmul(hidden_states, tf.transpose(miniBatch)), numberOfConfigurations)
term2 = tf.div(tf.matmul(hidden_probabilities2, tf.transpose(visible_states)), numberOfConfigurations)

gradient = tf.subtract(term1, term2)
# End implementation of CD1


momentumSpeedUpdated = tf.add(tf.multiply(tf.constant(momentumSpeedConstant, dtype=numberType), momentumSpeed), gradient)
weightsUpdated = tf.add(weights, tf.multiply(momentumSpeedUpdated, tf.constant(learningRate, dtype=numberType)))

updateMomentumSpeed = momentumSpeed.assign(momentumSpeedUpdated)
updateWeights = weights.assign(weightsUpdated)

for i in range(numberOfHiddenUnits):
    tf.summary.image('weights' + str(i), tf.reshape(updateWeights[i, :], [-1, 16, 16, 1]), 1)


tf.summary.histogram('momentumSpeedUpdated', momentumSpeedUpdated[0])
tf.summary.histogram('weightsUpdated',weightsUpdated[0])

merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    momentumSpeed.initializer.run()
    weights.initializer.run()

    for i in range(numberOfIterations):
        miniBatchValues = inputs[:, startOfNextMiniBatch: (startOfNextMiniBatch + miniBatchSize)]
        startOfNextMiniBatch = np.mod(startOfNextMiniBatch + miniBatchSize, inputs.shape[1])

        _, _, summary = sess.run([updateWeights, updateMomentumSpeed, merged_summary_op], feed_dict={miniBatch: miniBatchValues})

        if i % 1000 == 0:
            print("Iteration: ", i)
            writer.add_summary(summary, i)

    finalWeights = sess.run(weights)
    showWeights(finalWeights, (16, 16))

