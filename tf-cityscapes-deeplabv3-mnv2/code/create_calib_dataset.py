# Create Cityscapes Datasets for Calibration

import os
import shutil
import glob
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset_dir', type=str, 
    default=os.path.join('leftImg8bit_trainvaltest', 'leftImg8bit'),
    help='Top directory of Cityscapes dataset')


def main(dataset_dir):
    # Num. of calibration images
    num = 1000

    # Output directory
    out_dir_top = '_dataset'
    out_dir_calib = os.path.join(out_dir_top, 'calib')

    # Create output directories
    if os.path.exists(out_dir_top):
        shutil.rmtree(out_dir_top)
    os.mkdir(out_dir_top)

    if os.path.exists(out_dir_calib):
        shutil.rmtree(out_dir_calib)
    os.mkdir(out_dir_calib)

    # Load dataset
    train_dir = os.path.join(dataset_dir, 'train')
    if not os.path.exists(train_dir):
        print('Could not find train data directory: {}'.format(train_dir))
        return

    img_list = sorted(glob.glob(os.path.join(train_dir, '**', '*.png')))

    if num > len(img_list):
        print('Num. of calibration images must be <= num. of train images...')
        exit

    np.random.seed(123)
    idx = np.asarray(range(0, len(img_list)))
    np.random.shuffle(idx)

    for i in idx[0:num]:
        shutil.copy(img_list[i], out_dir_calib)


if __name__ == "__main__":
    FLAGS, _ = parser.parse_known_args()
    main(FLAGS.dataset_dir)