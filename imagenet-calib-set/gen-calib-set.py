# creating calibration dataset from ImageNet2012 validation set
# Based on: https://gist.github.com/ashutoshbsathe/27558207fd5f0bc6a769bef6ff8eb96a
import argparse
import os
import shutil

import tqdm
import yaml

# https://github.com/tensorflow/models/blob/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt
VALIDATION_SYNSET_LABELS = 'imagenet_2012_validation_synset_labels.txt'  # (above link)
# https://gist.githubusercontent.com/fnielsen/4a5c94eaa6dcdf29b7a62d886f540372/raw/d25516d26be4a8d3e0aeebe9275631754b8e2c73/imagenet_label_to_wordnet_synset.txt
LABEL_TO_SYNSET_MAP_FILE = 'imagenet_label_to_wordnet_synset.txt'  # (above link)


def main(src_dir: str, dst_dir: str) -> None:
    """main function.
    """
    with open(LABEL_TO_SYNSET_MAP_FILE, 'r') as f:
        labels_synset_json = f.read().replace('\n', ' ')
    labels_synset = yaml.safe_load(labels_synset_json)
    synset_to_label_dict = {}
    for k, v in labels_synset.items():
        synset_to_label_dict['n' + v['id'].split('-')[0]] = k
    with open(VALIDATION_SYNSET_LABELS, 'r') as f:
        lines = f.readlines()
    synset_to_label = list()
    for synset in lines:
        key = synset.replace('\n', '')
        synset_to_label.append(synset_to_label_dict.get(key, '134'))

    for i, cls in tqdm.tqdm(enumerate(synset_to_label, start=1)):
        src_path = os.path.join(src_dir, f'ILSVRC2012_val_{i:08d}.JPEG')
        dst_dir2 = os.path.join(dst_dir, f'{cls}')
        dst_path = os.path.join(dst_dir2, f'ILSVRC2012_val_{i:08d}.JPEG')
        os.makedirs(dst_dir2, exist_ok=True)
        shutil.copyfile(src_path, dst_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'src_dir', type=str,
        default='val', help='path to validation dataset directory')
    parser.add_argument(
        'dst_dir', type=str,
        default='val', help='path to validation dataset directory')
    args = parser.parse_args()
    main(args.src_dir, args.dst_dir)
