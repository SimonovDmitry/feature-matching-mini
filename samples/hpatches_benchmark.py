import argparse
import sys
import logging
from pathlib import Path

from src.detectors import Detector
from src.descriptors import Descriptor
from src.matchers import Matcher, OpenCVMatcher
from src.feature_matcher import FeatureMatcherCV2

from samples.utils import build_config
from samples.hpatches_utils import HPatchesDataManager, HPatchesTask


logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger("hpatches_benchmark")


def parser():
    arg_parser = argparse.ArgumentParser(
        description="HPatches Benchmark Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    available_detectors = list(Detector._METHODS.keys())
    available_descriptors = list(Descriptor._METHODS.keys())
    available_matchers = list(Matcher._METHODS.keys())
    available_matchers_modes = list(OpenCVMatcher._MODE)
    available_task = list(HPatchesTask._TASKS.keys())
    available_devices = ['cpu', 'cuda', 'mps']

    arg_parser.add_argument('-det', '--detector', type=str, default='sift',
                            choices=available_detectors, help='Detector algorithm')
    arg_parser.add_argument('-des', '--descriptor', type=str, default='sift',
                            choices=available_descriptors, help='Descriptor algorithm')
    arg_parser.add_argument('-mat', '--matcher', type=str, default='bf',
                            choices=available_matchers, help='Matching algorithm')

    arg_parser.add_argument('-d', '--device', type=str, default=None,
                            choices=available_devices, help='The device on which the script will be run')
    arg_parser.add_argument('-t', '--task', type=str, default='matching',
                            choices=available_task, help='Descriptor algorithm')
    arg_parser.add_argument('-p', '--path', type=Path, required=True,
                            help='Path to hpatches-release folder')
    arg_parser.add_argument('-n', '--num-scenes', type=int, default=None,
                            help='Number of scenes to process (default: all)')
    arg_parser.add_argument('-pt', '--pixel-threshold', type=float, default=5.0,
                           help='Pixel threshold for homography verification')

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
    return arg_parser.parse_args()



def main():
    args = parser()

    try:
        if not args.path.exists():
            logger.error(f"Dataset path does not exist: {args.path}")
            return 1

        logger.info("Starting HPatches Benchmark Pipeline")

        config = build_config(args)
        logger.info(f"Config: {config}")

        feature_matcher = FeatureMatcherCV2(detector=args.detector, descriptor=args.descriptor,
                                            matcher=args.matcher, logger=logger, config=config)
        dm = HPatchesDataManager(raw_data_path=args.path, logger=logger)
        dataset = dm.load_dataset(num_scenes=args.num_scenes)
        matching_data = {scene: {} for scene in dataset.keys()}

        for scene_name, data in dataset.items():
            img_ref = data['ref_img']

            for i, target in data['targets'].items():
                img_tgt = target['image']
                H = target['H']

                features_ref, features_tgt, correspondences = feature_matcher.match(img_ref, img_tgt)
                matching_data[scene_name][i] = {
                    'kp_ref': features_ref['kp'],
                    'kp_tgt': features_tgt['kp'],
                    'matches': correspondences['matches'],
                    'H': H
                }

        task = HPatchesTask.create(task_name=args.task, logger=logger, pixel_threshold=args.pixel_threshold)
        results = task.eval_task(matching_data, list(dataset.keys()))
        task.report_metrics(results, "End-to-End Pipeline")

        logger.info("Pipeline finished successfully")
        return 0

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main() or 0)
