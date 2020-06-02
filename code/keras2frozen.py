"""
Freeze Keras .h5 model for quantization
"""

import os
import argparse
import tempfile
import tensorflow as tf
from tensorflow.python.tools import freeze_graph


parser = argparse.ArgumentParser()
parser.add_argument(
    '--keras_model', type=str, default='',
    help='Keras model file (*.h5)')


# keras_model_path: Input Keras model (complete .h5 file)
def main(keras_model_path):
    if not os.path.exists(keras_model_path):
        print('Could not fine input Keras model file: {} '.format(keras_model_path))
        return

    # Load model
    tf.keras.backend.set_learning_phase(0)
    model = tf.keras.models.load_model(keras_model_path)
    # model.summary()

    # Model info (input/output nodes)
    input_nodes = ','.join([out.op.name for out in model.inputs])
    output_nodes = ','.join([out.op.name for out in model.outputs])
    print('Model: {}'.format(model.name))
    print('  Input node name: {}'.format(input_nodes))
    print('  Output node name: {}'.format(output_nodes))

    # Generate checkpoint via saver model
    #  Conversion via estimator decreases model accuracy??
    saver = tf.train.Saver()

    tf_session = tf.keras.backend.get_session()

    with tempfile.TemporaryDirectory() as tempdir:
        ckpt = os.path.join(tempdir, model.name + '.ckpt')
        _ = saver.save(tf_session, ckpt)

        _ = freeze_graph.freeze_graph(
            None, None, True, ckpt,
            output_nodes, None, None, model.name + '_frozen.pb', True, None, input_meta_graph=ckpt + '.meta')


if __name__ == '__main__':
    FLAGS, _ = parser.parse_known_args()
    main(FLAGS.keras_model)
