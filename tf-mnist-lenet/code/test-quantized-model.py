# Testing Quantized Model

import os
import numpy as np
import tensorflow as tf

print('TensorFlow Version: %s' % tf.__version__)

# Quantized model
model_path = os.path.join('quantized', 'quantize_eval_model.pb')
if not os.path.exists(model_path):
    print('Quantized model %s cannot be found...' % model_path)

# Need to run this before importing quantized graph
tf.contrib.resampler

# Load model
graph_def = tf.GraphDef()
graph_def.ParseFromString(tf.gfile.GFile(model_path, "rb").read())

graph = tf.Graph()
graph.as_default()

tf.import_graph_def(graph_def, name="")

input_tensor = tf.get_default_graph().get_tensor_by_name('conv2d_input:0')
output_tensor = tf.get_default_graph().get_tensor_by_name('dense_1/Softmax:0')

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# x = x_train/255.
# y = y_train
x = x_test/255.
y = y_test

# Evaluate - Use tf.Session
batch_size = 100
iter = int(len(x)/batch_size)

result = np.zeros(len(x))

with tf.Session() as sess:
    for i in range(0, iter):
        img = np.expand_dims(x[batch_size*i:batch_size*(i+1)], axis=3)

        feed_dict = {input_tensor: img}
        out_0 = sess.run([output_tensor], feed_dict)

        result[batch_size*i:batch_size*(i+1)] = np.argmax(out_0[0], axis=1)

print('Accuracy: %d / %d' % (np.sum(result == y), len(x)))
