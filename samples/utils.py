import argparse
from pathlib import Path
from src.detectors import Detector
from src.descriptors import Descriptor
from src.matchers import Matcher, OpenCVMatcher
from src.algorithms import DNN_DETECTORS, DNN_DESCRIPTORS, DNN_MATCHERS, OPENCV_MATCHERS
from src.super_point import SuperPoint # noqa: F401
from src.lightglue_matcher import LightGlue # noqa: F401
from src.lightglue_pipeline import LightGlueFeatureExtractor # noqa: F401


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

def performance_tests_parser():
    arg_parser = argparse.ArgumentParser(
        description="Feature matching staged performance test",
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
