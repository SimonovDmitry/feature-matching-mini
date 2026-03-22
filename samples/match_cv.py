import argparse
import sys
import logging
from pathlib import Path

from src.detectors import Detector
from src.descriptors import Descriptor
from src.matchers import Matcher
from src.feature_matcher import FeatureMatcherCV2
from src.image_utils import read_image, save_image, show_image


logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger("CV_Sample")


def parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(
        description="Matching points in two images using OpenCV algorithms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    available_detectors = list(Detector._METHODS.keys())
    available_descriptors = list(Descriptor._METHODS.keys())
    available_matchers = list(Matcher._METHODS.keys())
    available_matchers_modes = list(Matcher._MODES.keys())

    arg_parser.add_argument('-det', '--detector', type=str, default='sift',
                            choices=available_detectors, help='Detector algorithm')

    arg_parser.add_argument('-des', '--descriptor', type=str, default='sift',
                            choices=available_descriptors, help='Descriptor algorithm')

    arg_parser.add_argument('-mat', '--matcher', type=str, default='bf',
                            choices=available_matchers, help='Matching algorithm')

    arg_parser.add_argument('-mat_m', '--matcher_mode', type=str, default='simple',
                            choices=available_matchers_modes, help='Matching mode')

    arg_parser.add_argument('-i1', '--image1', type=Path, required=True,
                            help='Path to the first image')
    arg_parser.add_argument('-i2', '--image2', type=Path, required=True,
                            help='Path to the second image')

    arg_parser.add_argument('-s', '--save', type=Path, default='',
                            help='Path to save result image')
    arg_parser.add_argument('-v', '--show', action='store_true',
                           help='Show the matching result in a window')

    return arg_parser.parse_args()


def main() -> int:
    args = parser()

    try:
        if not args.image1.exists() or not args.image2.exists():
            logger.error(f"One of the images does not exist: {args.image1} or {args.image2}")
            return 1

        logger.info("Starting Feature Matching Pipeline")
        logger.info(f"Comparing pair: {args.image1} and {args.image2}")

        img1 = read_image(str(args.image1))
        img2 = read_image(str(args.image2))


        feature_matcher = FeatureMatcherCV2(detector=args.detector, descriptor=args.descriptor,
            matcher=args.matcher, matcher_mode=args.matcher_mode, logger=logger)

        kp1, kp2, matches = feature_matcher.match(img1, img2)
        res_img = feature_matcher.visualize_matches(img1, kp1, img2, kp2, matches)

        if args.save:
            save_image(res_img, save_path=str(args.save))
            logger.info(f"Result successfully saved to: {args.save}")

        if args.show:
            show_image(res_img, title="Matched Image")

        logger.info("Pipeline finished successfully")
        return 0

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main() or 0)
