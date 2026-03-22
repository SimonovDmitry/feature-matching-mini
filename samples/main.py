import sys
import argparse
from pathlib import Path
import logging
import cv2 as cv
from descriptors import Descriptors
from matchers import Matchers

logging.basicConfig(level=logging.INFO, force=True)


def cli_argument_parser():
    parser = argparse.ArgumentParser(
        prog='Feature matching',
        description='Draw key points and matches'
    )
    parser.add_argument('-i1', '--image1',
                        required=True,
                        help='Path to an image 1',
                        type=str,
                        dest='img1')
    parser.add_argument('-i2', '--image2',
                        required=True,
                        help='Path to an image 2',
                        type=str,
                        dest='img2')
    parser.add_argument('--method',
                        required=True,
                        choices=['sift', 'orb'],
                        help='Selecting a method',
                        type=str)
    parser.add_argument('--matcher',
                        required=True,
                        choices=['bf', 'flann'],
                        help='Selecting a matcher',
                        type=str)
    parser.add_argument('--knn',
                        action='store_true',
                        help='Using knn matcher')
    args = parser.parse_args()
    return args


def image_read(image):
    if image is None:
        raise ValueError('Empty path to the image')
    filepath = Path(image)
    if not filepath.exists():
        raise ValueError('Incorrect path to the image')
    src_image = cv.imread(image)
    return src_image


def main():
    args = cli_argument_parser()
    try:
        image1 = image_read(args.img1)
        image2 = image_read(args.img2)
        method_instance = Descriptors()
        kp1, des1, kp2, des2 = method_instance.apply_method(image1, image2, args.method)
        matcher_instance = Matchers()
        result = matcher_instance.apply_matcher(image1, image2, kp1,
                                                kp2, des1, des2, args.method,
                                                args.matcher, args.knn)
        cv.imshow("image", result)
        cv.waitKey(0)
        cv.destroyAllWindows()
    except Exception as e:
        logging.error(e)
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main() or 0)
