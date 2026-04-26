from src.algorithms import DNN_DETECTORS, DNN_DESCRIPTORS, DNN_MATCHERS, OPENCV_MATCHERS


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
