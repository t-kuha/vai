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

    # Quantized model
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

    input_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('input_1:0')
    output_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('fc10/Softmax:0')

    # Load dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # x = x_train
    # y = y_train
    x = x_test
    y = y_test

    # Normalize per image
    m = np.mean(x, axis=(1, 2, 3))
    s = np.std(x, axis=(1, 2, 3))
    x = (x - m[:, np.newaxis, np.newaxis, np.newaxis])/s[:, np.newaxis, np.newaxis, np.newaxis]

    # Evaluate - Use tf.Session
    batch_size = 100
    iter = int(len(x)/batch_size)

    result = np.zeros(len(x))

    with tf.compat.v1.Session() as sess:
        for i in range(0, iter):
            img = x[batch_size*i:batch_size*(i+1)]
            feed_dict = {input_tensor: img}
            out_0 = sess.run([output_tensor], feed_dict)

            result[batch_size*i:batch_size*(i+1)] = np.argmax(out_0[0], axis=1)

    num_correct = np.sum(result.flatten() == y.flatten())
    print('Accuracy: %f [%%] (%d / %d)' % (num_correct/len(x)*100., num_correct, len(x)))


if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS.model_path)
