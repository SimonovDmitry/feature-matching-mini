import argparse
import sys
import logging
from pathlib import Path

from src.image_utils import read_image, save_image, show_image
from src.algorithms import OPENCV_MATCHERS_MODE, NEURAL_MATCHERS

from src.detectors import Detector
from src.descriptors import Descriptor
from src.matchers import Matcher
from src.feature_matcher import FeatureMatcherCV2


logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger("CV_Sample")


def build_detector_config(args):
    config = {}
    if args.det_nfeatures is not None:
        config['nfeatures'] = args.det_nfeatures
    if args.det_noctave is not None:
        config['nOctaveLayers'] = args.det_noctave
    if args.det_threshold is not None:
        config['threshold'] = args.det_threshold
    if args.det_device is not None:
        config['device'] = args.det_device
    return config


def build_descriptor_config(args):
    config = {}
    if args.des_nfeatures is not None:
        config['nfeatures'] = args.des_nfeatures
    if args.des_threshold is not None:
        config['threshold'] = args.des_threshold
    if args.des_scale is not None:
        config['scale_factor'] = args.des_scale
    if args.des_device is not None:
        config['device'] = args.des_device
    return config


def build_matcher_config(args):
    config = {}
    if args.matcher not in NEURAL_MATCHERS:
        config['mode'] = args.matcher_mode
    if args.mat_ratio is not None:
        config['ratio'] = args.mat_ratio
    if args.mat_device is not None:
        config['device'] = args.mat_device
    if args.mat_cross_check is not None:
        config['cross_check'] = args.mat_cross_check
    return config


def build_preprocessor_config(args):
    config = {}
    if args.pre_device is not None:
        config['device'] = args.pre_device
    return config


def build_config(args):
    return {
        'detector': build_detector_config(args),
        'descriptor': build_descriptor_config(args),
        'matcher': build_matcher_config(args),
        'preprocessor': build_preprocessor_config(args),
    }


def parser():
    arg_parser = argparse.ArgumentParser(
        description="Matching points in two images using OpenCV algorithms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    available_detectors = list(Detector._METHODS.keys())
    available_descriptors = list(Descriptor._METHODS.keys())
    available_matchers = list(Matcher._METHODS.keys())
    available_matchers_modes = list(OPENCV_MATCHERS_MODE)
    available_devices = ['cpu', 'cuda', 'mps']

    arg_parser.add_argument('-det', '--detector', type=str, default='sift',
                            choices=available_detectors, help='Detector algorithm')
    arg_parser.add_argument('-des', '--descriptor', type=str, default='sift',
                            choices=available_descriptors, help='Descriptor algorithm')
    arg_parser.add_argument('-mat', '--matcher', type=str, default='bf',
                            choices=available_matchers, help='Matching algorithm')

    arg_parser.add_argument('-i1', '--image1', type=Path, required=True,
                            help='Path to the first image')
    arg_parser.add_argument('-i2', '--image2', type=Path, required=True,
                            help='Path to the second image')

    arg_parser.add_argument('-s', '--save', type=Path, default=None,
                            help='Path to save result image')
    arg_parser.add_argument('-v', '--show', action='store_true',
                            help='Show the matching result in a window')

    det_group = arg_parser.add_argument_group('Detector config')
    det_group.add_argument('-dn', '--det-nfeatures', type=int, default=None,
                           help='Max number of features to detect')
    det_group.add_argument('-do', '--det-noctave', type=int, default=None,
                           help='Number of octave layers')
    det_group.add_argument('-dt', '--det-threshold', type=float, default=None,
                           help='Detection threshold')
    det_group.add_argument('-dd', '--det-device', type=str, default=None,
                           choices=available_devices, help='Device for detector')

    des_group = arg_parser.add_argument_group('Descriptor config')
    des_group.add_argument('-dsen', '--des-nfeatures', type=int, default=None,
                           help='Max number of features for descriptor')
    des_group.add_argument('-dsdt', '--des-threshold', type=float, default=None,
                           help='Descriptor threshold')
    des_group.add_argument('-dss', '--des-scale', type=float, default=None,
                           help='Scale factor')
    des_group.add_argument('-dsd', '--des-device', type=str, default=None,
                           choices=available_devices, help='Device for descriptor')

    mat_group = arg_parser.add_argument_group('Matcher config')
    mat_group.add_argument('-mat_m', '--matcher_mode', type=str, default='simple',
                            choices=available_matchers_modes, help='Matching mode')
    mat_group.add_argument('-mr', '--mat-ratio', type=float, default=None,
                           help='Ratio threshold for KNN')
    mat_group.add_argument('-md', '--mat-device', type=str, default=None,
                           choices=available_devices, help='Device for matcher')
    mat_group.add_argument('-mc', '--mat-cross-check', action='store_true', default=None,
                           help='Enable cross-check for BF matcher')

    pre_group = arg_parser.add_argument_group('Preprocessor config')
    pre_group.add_argument('-pd', '--pre-device', type=str, default=None,
                           choices=available_devices, help='Device for preprocessor')


    return arg_parser.parse_args()


def main():
    args = parser()

    try:
        if not args.image1.exists() or not args.image2.exists():
            logger.error(f"One of the images does not exist: {args.image1} or {args.image2}")
            return 1

        logger.info("Starting Feature Matching Pipeline")
        logger.info(f"Comparing pair: {args.image1} and {args.image2}")

        config = build_config(args)
        logger.info(f"Config: {config}")

        img1 = read_image(args.image1)
        img2 = read_image(args.image2)

        feature_matcher = FeatureMatcherCV2(detector=args.detector, descriptor=args.descriptor,
                                            matcher=args.matcher, logger=logger, config=config)
        features0, features1, correspondences = feature_matcher.match(img1, img2)
        res_img = feature_matcher.visualize_matches(img1, features0, img2, features1, correspondences)

        if args.save:
            save_image(res_img, save_path=args.save)
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