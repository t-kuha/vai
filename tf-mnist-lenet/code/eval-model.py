# Testing Frozen (including quantized) Model

import os
import numpy as np
import argparse
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_path', type=str,
    default=os.path.join('quantized', 'quantize_eval_model.pb'),
    help='TensorFlow frozem mode file (*.pb)')


def main(model_path):
    print('TensorFlow Version: %s' % tf.__version__)

    if not os.path.exists(model_path):
        print('Quantized model %s cannot be found...' % model_path)

    # Need to run this before importing quantized graph
    tf.contrib.resampler

    # Load model
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(tf.io.gfile.GFile(model_path, "rb").read())

    graph = tf.Graph()
    graph.as_default()

    tf.import_graph_def(graph_def, name="")

    input_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('conv2d_input:0')
    output_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('dense_1/Softmax:0')

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

    with tf.compat.v1.Session() as sess:
        for i in range(0, iter):
            img = np.expand_dims(x[batch_size*i:batch_size*(i+1)], axis=3)

            feed_dict = {input_tensor: img}
            out_0 = sess.run([output_tensor], feed_dict)

            result[batch_size*i:batch_size*(i+1)] = np.argmax(out_0[0], axis=1)

    print('Accuracy: %d / %d' % (np.sum(result == y), len(x)))


if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS.model_path)
