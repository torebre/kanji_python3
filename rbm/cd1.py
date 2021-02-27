import tensorflow as tf


# This is the gradient function
def cd1(inputs, rbm_weights):
    # Start implementation of contrastive divergence with 1 step (CD1)
    visible_data_sampled = tf.cast(tf.greater_equal(inputs, tf.random_uniform(inputs.shape)), "float32")

    # Visible state to hidden probabilities
    ones = tf.constant(1, "float32", [rbm_weights.shape[0], visible_data_sampled.shape[1]])
    multiply_term = tf.exp(tf.matmul(tf.negative(rbm_weights), visible_data_sampled))

    hidden_probabilities = tf.div(ones, tf.add(ones, multiply_term))
    hidden_states = tf.cast(tf.greater_equal(hidden_probabilities, tf.random_uniform(hidden_probabilities.shape)),
                            "float32")

    # Compute the hidden state to visible probabilities
    ones_hidden = tf.constant(1, "float32", inputs.shape)
    multiply_term_hidden = tf.exp(tf.matmul(tf.transpose(tf.negative(rbm_weights)), hidden_states))
    visible_probabilities = tf.div(ones_hidden, tf.add(ones_hidden, multiply_term_hidden))

    visible_states = tf.cast(tf.greater_equal(visible_probabilities, tf.random_uniform(inputs.shape)), "float32")

    hidden_probabilities2 = tf.div(ones, tf.add(ones, tf.exp(tf.matmul(tf.negative(rbm_weights), visible_states))))

    numberOfConfigurations = tf.constant(inputs.shape[1].value, dtype="float32")

    term1 = tf.div(tf.matmul(hidden_states, tf.transpose(inputs)), numberOfConfigurations)
    term2 = tf.div(tf.matmul(hidden_probabilities2, tf.transpose(visible_states)), numberOfConfigurations)

    return tf.subtract(term1, term2)
