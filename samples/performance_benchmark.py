import argparse
import sys
import logging
import numpy as np
from pathlib import Path

from src.image_utils import read_image
from src.preprocessor import Preprocessor
from src.algorithms import (DETECTOR_DESCRIPTOR_COMPATIBILITY, DESCRIPTOR_MATCHER_COMPATIBILITY, DNN_MATCHERS,
                            OPENCV_MATCHERS, DNN_ALGORITHMS)
from src.detectors import Detector
from src.descriptors import Descriptor
from src.matchers import Matcher, OpenCVMatcher
from src.feature_matcher import FeatureMatcherCV2
from src.performance_profiler import PerformanceProfiler


logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger("CV_Sample")


def build_detector_config(args):
    config = dict()
    if (args.device is not None) and (args.detector in DNN_DETECTORS):
        config['device'] = args.device
    if args.det_nfeatures is not None:
        config['nfeatures'] = args.det_nfeatures
    if args.det_noctave is not None:
        config['nOctaveLayers'] = args.det_noctave
    if args.det_threshold is not None:
        config['threshold'] = args.det_threshold
    return config


def build_descriptor_config(args):
    config = dict()
    if (args.device is not None) and (args.descriptor in DNN_DESCRIPTORS):
        config['device'] = args.device
    if args.des_nfeatures is not None:
        config['nfeatures'] = args.des_nfeatures
    if args.des_threshold is not None:
        config['threshold'] = args.des_threshold
    if args.des_scale is not None:
        config['scale_factor'] = args.des_scale
    return config


def build_matcher_config(args):
    config = dict()
    if (args.device is not None) and (args.matcher in DNN_MATCHERS):
        config['device'] = args.device
    if args.matcher in OPENCV_MATCHERS:
        config['mode'] = args.matcher_mode
    if args.mat_ratio is not None:
        config['ratio'] = args.mat_ratio
    if args.mat_cross_check is not None:
        config['cross_check'] = args.mat_cross_check
    return config


def build_preprocessor_config(args):
    config = dict()
    if args.device is not None:
        config['device'] = args.device
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
        description="Feature matching performance test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    available_detectors = list(Detector._METHODS.keys())
    available_descriptors = list(Descriptor._METHODS.keys())
    available_matchers = list(Matcher._METHODS.keys())
    available_matchers_modes = list(OpenCVMatcher._MODE)
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

    arg_parser.add_argument('-d', '--device', type=str, default=None,
                            choices=available_devices, help='The device on which the script will be run')

    det_group = arg_parser.add_argument_group('Detector config')
    det_group.add_argument('-dn', '--det-nfeatures', type=int, default=None,
                           help='Max number of features to detect')
    det_group.add_argument('-do', '--det-noctave', type=int, default=None,
                           help='Number of octave layers')
    det_group.add_argument('-dt', '--det-threshold', type=float, default=None,
                           help='Detection threshold')

    des_group = arg_parser.add_argument_group('Descriptor config')
    des_group.add_argument('-dsen', '--des-nfeatures', type=int, default=None,
                           help='Max number of features for descriptor')
    des_group.add_argument('-dsdt', '--des-threshold', type=float, default=None,
                           help='Descriptor threshold')
    des_group.add_argument('-dss', '--des-scale', type=float, default=None,
                           help='Scale factor')

    mat_group = arg_parser.add_argument_group('Matcher config')
    mat_group.add_argument('-mat_m', '--matcher_mode', type=str, default='simple',
                           choices=available_matchers_modes, help='Matching mode')
    mat_group.add_argument('-mr', '--mat-ratio', type=float, default=None,
                           help='Ratio threshold for KNN')
    mat_group.add_argument('-mc', '--mat-cross-check', action='store_true', default=None,
                           help='Enable cross-check for BF matcher')
    arg_parser.add_argument('-n', '--iterations', type=int, default=10,
                            help='Number of iterations for performance testing (default: 10)')
    return arg_parser.parse_args()


def staged_performance_test(logger, profiler, preprocessor, detector, detector_name, descriptor, descriptor_name,
                         matcher, matcher_name, img0, img1, iterations):
    times_detectors = []
    times_descriptors = []
    times_detectors_and_descriptors = []
    times_matchers = []

    if detector_name in DNN_ALGORITHMS or descriptor_name in DNN_ALGORITHMS:
        for _ in range(iterations):
            res_extract_dict0, time_desc0 = profiler.profile_dnn_extractor(descriptor, img0)
            res_extract_dict1, time_desc1 = profiler.profile_dnn_extractor(descriptor, img1)
            times_detectors_and_descriptors.append(time_desc0)
            times_detectors_and_descriptors.append(time_desc1)
            features0 = preprocessor.prepare_features(res_extract_dict0, from_algo=descriptor_name, to_algo=matcher_name)
            features1 = preprocessor.prepare_features(res_extract_dict1, from_algo=descriptor_name, to_algo=matcher_name)

            res_match_dict, time_match = profiler.profile_matching(matcher, features0, features1)
            times_matchers.append(time_match)

        min_time_detectors_and_descriptors = np.min(times_detectors_and_descriptors)
        mean_time_detectors_and_descriptors = np.mean(times_detectors_and_descriptors)

        min_time_matcher = np.min(times_matchers)
        mean_time_matcher = np.mean(times_matchers)
        logger.info(f"Min time feature extract: {min_time_detectors_and_descriptors}\n"
                    f"Mean time feature extract: {mean_time_detectors_and_descriptors}\n"
                    f"Min time match: {min_time_matcher}\n"
                    f"Mean time match: {mean_time_matcher}\n")
    else:
        for _ in range(iterations):
            res_detect_dict0, time_detect0 = profiler.profile_detection(detector, img0)
            res_detect_dict1, time_detect1 = profiler.profile_detection(detector, img1)
            times_detectors.append(time_detect0)
            times_detectors.append(time_detect1)


            kp0 = preprocessor.prepare_features(res_detect_dict0, from_algo=detector_name, to_algo=descriptor_name)
            kp1 = preprocessor.prepare_features(res_detect_dict1, from_algo=detector_name, to_algo=descriptor_name)


            res_desc_dict0, time_desc0 = profiler.profile_descriptor(descriptor, img0, kp0)
            res_desc_dict1, time_desc1 = profiler.profile_descriptor(descriptor, img1, kp1)
            times_descriptors.append(time_desc0)
            times_descriptors.append(time_desc1)


            features0 = preprocessor.prepare_features(res_desc_dict0, from_algo=descriptor_name, to_algo=matcher_name)
            features1 = preprocessor.prepare_features(res_desc_dict1, from_algo=descriptor_name, to_algo=matcher_name)


            res_match_dict, time_match = profiler.profile_matching(matcher, features0, features1)
            times_matchers.append(time_match)


        min_time_detector = np.min(times_detectors)
        mean_time_detector = np.mean(times_detectors)


        min_time_descriptor = np.min(times_descriptors)
        mean_time_descriptor = np.mean(times_descriptors)


        min_time_matcher = np.min(times_matchers)
        mean_time_matcher = np.mean(times_matchers)


        logger.info(f"Min time detection: {min_time_detector}\n"
                    f"Mean time detection: {mean_time_detector}\n"
                    f"Min time descriptor: {min_time_descriptor}\n"
                    f"Mean time descriptor: {mean_time_descriptor}\n"
                    f"Min time match: {min_time_matcher}\n"
                    f"Mean time match: {mean_time_matcher}\n")


def pipeline_performance_test(logger, profiler, detector, detector_name, descriptor, descriptor_name,
                         matcher, matcher_name, img0, img1, iterations):
    times = []

    for _ in range(iterations):
        if detector_name in DNN_ALGORITHMS or descriptor_name in DNN_ALGORITHMS:
            res, time = profiler.profile_dnn_pipeline(descriptor, descriptor_name, matcher, matcher_name, img0, img1)
        else:
            res, time = profiler.profile_pipeline(detector, detector_name, descriptor, descriptor_name, matcher, matcher_name, img0, img1)
        times.append(time)
    min_time = np.min(times)
    mean_time = np.mean(times)
    logger.info(f"Min time pipeline test: {min_time}\n"
                f"Mean time pipeline test: {mean_time}")


def main():
    args = parser()

    try:
        if not args.image1.exists() or not args.image2.exists():
            logger.error(f"One of the images does not exist: {args.image1} or {args.image2}")
            return 1

        config = build_config(args)
        logger.info(f"Config: {config}")

        img0 = read_image(args.image1)
        img1 = read_image(args.image2)

        detector = Detector.create(detector_name=args.detector, logger=args.logger, config=args.detector_config)
        descriptor = Descriptor.create(descriptor_name=args.descriptor, logger=args.logger,
                                       config=args.descriptor_config)
        matcher = Matcher.create(matcher_name=args.matcher, descriptor_name=descriptor,
                                 logger=args.logger, config=args.matcher_config)
        preprocessor = Preprocessor(config=args.preprocessor_config, logger=args.logger)

        img0 = preprocessor.prepare_image(img0, from_algo='opencv', to_algo=args.detector)
        img1 = preprocessor.prepare_image(img1, from_algo='opencv', to_algo=args.etector)

        profiler = PerformanceProfiler(preprocessor)

        iterations = args.iterations
        logger.info(f"Running performance test with {iterations} iterations...")

        staged_performance_test(
            logger=logger,
            profiler=profiler,
            preprocessor=preprocessor,
            detector=detector,
            detector_name=args.detector,
            descriptor=descriptor,
            descriptor_name=args.descriptor,
            matcher=matcher,
            matcher_name=args.matcher,
            img0=img0,
            img1=img1,
            iterations=iterations
        )

        pipeline_performance_test(
            logger=logger,
            profiler=profiler,
            detector=detector,
            detector_name=args.detector,
            descriptor=descriptor,
            descriptor_name=args.descriptor,
            matcher=matcher,
            matcher_name=args.matcher,
            img0=img0,
            img1=img1,
            iterations=iterations
        )

        return 0

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main() or 0)