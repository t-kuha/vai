# Create MNIST Datasets for Calibration & Test

import os
import shutil
import cv2
import numpy as np
import tensorflow as tf


print('TensorFlow Version: %s' % tf.__version__)

# Num. of calibration images
num = 1000

# Output directory
out_dir_top = '_dataset'
out_dir_calib = os.path.join(out_dir_top, 'calib')
out_dir_test = os.path.join(out_dir_top, 'test')

# Create fresh output directories
if os.path.exists(out_dir_top):
    shutil.rmtree(out_dir_top)
os.mkdir(out_dir_top)

if os.path.exists(out_dir_calib):
    shutil.rmtree(out_dir_calib)
os.mkdir(out_dir_calib)

if os.path.exists(out_dir_test):
    shutil.rmtree(out_dir_test)
os.mkdir(out_dir_test)


# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Save as .png file
idx = np.asarray(range(0, len(x_train)))
np.random.shuffle(idx)  # In-place operation

with open('_calib.txt', 'w') as f:
    for i in idx[0:num]:
        fname = '{}.png'.format(i)
        cv2.imwrite(os.path.join(out_dir_calib, fname), x_train[i])
        f.write('{} {}\n'.format(fname, y_train[i]))

with open('_test.txt', 'w') as f:
    for i, data in enumerate(x_test):
        fname = '{}.png'.format(i)
        cv2.imwrite(os.path.join(out_dir_test, fname), data)
        f.write('{} {}\n'.format(fname, y_test[i]))
