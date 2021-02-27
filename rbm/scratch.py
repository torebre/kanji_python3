import sys
sys.path.extend(['/home/student/workspace/kanji_python/rbm'])

import tensorflow as tf
from cd1 import cd1


visible_data = tf.constant([0.4, 0.2, 0.6, 0.3])
rbm_weights = tf.constant([[0.5, 0.3, 0.8, 0.7], [0.2, 0.3, 0.2, 0.8]])
cd1(visible_data, rbm_weights)