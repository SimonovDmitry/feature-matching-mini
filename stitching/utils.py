import os
import cv2

from samples.utils import build_feature_matcher_config


def read_images(img_dir):
    imgs = []
    files = [os.path.join(img_dir, f)
             for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    files = sorted(files)
    for file in files:
        print(file)
        img = cv2.imread(file)
        if img is None:
            raise Exception(f'Error when reading image {file}')
        imgs.append(img)
    return imgs


def build_stitch_config(args):
    return {
        **build_feature_matcher_config(args)
    }
