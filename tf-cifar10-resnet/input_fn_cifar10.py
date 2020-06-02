import cv2
import os
import glob
import numpy as np

calib_image_dir = os.path.join('_dataset', 'calib')
calib_batch_size = 16

img_list = sorted(glob.glob(os.path.join(calib_image_dir, '*.png')))


def calib_input(iter):
    images = []

    # print('iter = %d' % iter)

    for index in range(0, calib_batch_size):
        idx_ = (iter * calib_batch_size + index) % len(img_list)
        path = img_list[idx_]

        # read image
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocessing (per-image normalization)
        m = np.mean(image)
        s = np.std(image)
        image = (image - m) / s

        images.append(image)

    return {"input_1": images}
