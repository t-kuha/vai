import os
import glob
import numpy as np
import cv2


calib_image_dir = os.path.join('_dataset', 'calib')
calib_batch_size = 1

img_list = sorted(glob.glob(os.path.join(calib_image_dir, '*.png')))


def calib_input(iter):
    images = []

    print('iter = %d' % iter)

    for index in range(0, calib_batch_size):
        idx_ = (iter * calib_batch_size + index) % len(img_list)
        path = img_list[idx_]

        # Load image & resize it
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(np.asarray(img), (2048, 1024))
        img = img.astype(np.float32) / 127.5 - 1

        # Final image
        image = np.zeros((1025, 2049, 3), np.float32)
        image[0:1024, 0:2048, :] = img

        images.append(image)

    return {"ImageTensor": images}
