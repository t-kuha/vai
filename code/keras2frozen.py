"""
# freeze_graph \
# --input_checkpoint=cifar10-resnet/model.ckpt-71108 \
# --output_graph=cifar10-resnet.pb \
# --output_node_names='softmax_tensor' \
# --input_meta_graph=./cifar10-resnet/model.ckpt-71108.meta \
# --input_binary=true
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
    model = tf.keras.models.load_model(keras_model_path)
    # model.summary()

    # Model info (input/output nodes)
    input_nodes = model.input.name.split(':')[0]
    output_nodes = model.output.name.split(':')[0]

    print('Model: {}'.format(model.name))
    print('  Input node name: {}'.format(input_nodes))
    print('  Output node name: {}'.format(output_nodes))

    with tempfile.TemporaryDirectory() as tempdir:
        # Convert to checkpoint
        estimator_model = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=tempdir)

        # THen freeze the checkpoint file
        ckpt = os.path.join(tempdir, 'keras', 'keras_model.ckpt')
        _ = freeze_graph.freeze_graph(
            None, None, True, ckpt,
            output_nodes, None, None, model.name+'_frozen.pb', True, None, input_meta_graph=ckpt+'.meta')


if __name__ == '__main__':
    FLAGS, _ = parser.parse_known_args()
    main(FLAGS.keras_model)
