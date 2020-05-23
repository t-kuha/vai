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

# Create output directories
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

for i in idx[0:num]:
    cv2.imwrite(os.path.join(out_dir_calib, '%d.png' % (i)), x_train[i])

for i in range(0, len(x_test)):
    cv2.imwrite(os.path.join(out_dir_test, '%d.png' % (i)), x_test[i])
