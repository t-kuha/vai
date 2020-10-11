# Evaluate MobileNet V2 based model on Cityscapes Validation set

import os
import glob
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset_type', type=str,
    default='val',
    help='Dataset type ("train", "val", "test")')
parser.add_argument(
    '--dataset_dir', type=str,
    default=os.getcwd(),
    help='Directory containing Cityscapes dataset')
parser.add_argument(
    '--model_path', type=str,
    default=os.path.join('deeplabv3_mnv2_cityscapes_train', 'frozen_inference_graph.pb'),
    help='TensorFlow frozen mode file (*.pb)')


def main(FLAGS):
    dataset_type = FLAGS.dataset_type
    dataset_dir = FLAGS.dataset_dir
    model_path = FLAGS.model_path

    # Dataset directory
    img_dir_top = os.path.join(dataset_dir, 'leftImg8bit', dataset_type)
    if not os.path.exists(img_dir_top):
        print('Input image cannot be found...')
        return
    img_dir_list = glob.glob(os.path.join(img_dir_top, '*'))

    # Model to evaluate
    if not os.path.exists(model_path):
        print('Model file cannot be found...')
        return

    # Load model
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(tf.io.gfile.GFile(model_path, "rb").read())

    graph = tf.Graph()
    graph.as_default()

    tf.import_graph_def(graph_def, name="")

    input_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('ImageTensor:0')
    output_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('SemanticPredictions:0')

    ytrueT = tf.compat.v1.placeholder(tf.uint8, shape=[1024, 2048])
    ypredT = tf.compat.v1.placeholder(tf.uint8, shape=[1024, 2048])
    weights = tf.compat.v1.placeholder(tf.uint8, shape=[1024, 2048])
    iou, conf_mat = tf.compat.v1.metrics.mean_iou(ytrueT, ypredT, num_classes=19, weights=weights)

    miou_list = []
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())

        for d in img_dir_list:
            img_list = sorted(glob.glob(os.path.join(d, '*.png')))

            # Check the existence of corresponding ground truth directory
            gt_dir = d.replace('leftImg8bit', 'gtFine')
            if not os.path.exists(gt_dir):
                print('Ground truth cannot be found...')

            for img_path in img_list:
                # print('---- {}'.format(img_path))

                # Load image
                image = Image.open(img_path)
                image = image.convert('RGB')
                input_image = np.expand_dims(np.asarray(image), 0)

                # Run inference
                output = sess.run([output_tensor], {input_tensor: input_image})

                # Post-processing: convert raw output to segmentation output
                pred = output[0][0]

                # Load ignore label data
                _, fname = os.path.split(img_path)
                fname = fname.replace('leftImg8bit', 'gtFine_labelTrainIds')
                img_label = np.asarray(Image.open(os.path.join(gt_dir, fname)))
                img_weights = np.not_equal(img_label, 255).astype(np.float32)

                img_ids = img_label.copy()
                img_ids[img_label == 255] = 0

                # Must run conf_mat
                sess.run([conf_mat], feed_dict={ytrueT: img_ids, ypredT: pred, weights: img_weights})
                miou = sess.run([iou], feed_dict={ytrueT: img_ids, ypredT: pred, weights: img_weights})
                miou_list.append(miou)
                # print(miou)

    print('---- Mean IOU: {}'.format(np.mean(miou_list)))

if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
