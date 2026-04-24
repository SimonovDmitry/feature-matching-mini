import argparse
import sys
import logging
import numpy as np
import cv2 as cv
from pathlib import Path

from src.algorithms import DNN_DESCRIPTORS
from src.descriptors import Descriptor
from src.super_point import SuperPoint
from src.preprocessor import Preprocessor

from samples.hpatches_utils import HPatchesTask, HPatchesDataManager

logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger("hpatches_benchmark")


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


def build_preprocessor_config(args):
    config = dict()
    if args.device is not None:
        config['device'] = args.device
    return config


def build_config(args):
    return {
            'descriptor': build_descriptor_config(args),
            'preprocessor': build_preprocessor_config(args),
           }


def parser():
    arg_parser = argparse.ArgumentParser(
        description="HPatches Benchmark Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    available_descriptors = list(Descriptor._METHODS.keys())
    available_task = list(HPatchesTask._TASKS.keys())
    available_devices = ['cpu', 'cuda', 'mps']

    arg_parser.add_argument('-des', '--descriptor', type=str, default='sift',
                            choices=available_descriptors, help='Descriptor algorithm')
    arg_parser.add_argument('-t', '--task', type=str, default='matching',
                            choices=available_task, help='Descriptor algorithm')

    arg_parser.add_argument('-d', '--device', type=str, default=None,
                            choices=available_devices, help='The device on which the script will be run')
    arg_parser.add_argument('-p', '--path', type=Path, required=True,
                            help='Path to hpatches-release folder')
    arg_parser.add_argument('-n', '--num-scenes', type=int, default=None,
                            help='Number of scenes to process (default: all)')

    des_group = arg_parser.add_argument_group('Descriptor config')
    des_group.add_argument('-dsen', '--des-nfeatures', type=int, default=None,
                           help='Max number of features for descriptor')
    des_group.add_argument('-dsdt', '--des-threshold', type=float, default=None,
                           help='Descriptor threshold')
    des_group.add_argument('-dss', '--des-scale', type=float, default=None,
                           help='Scale factor')
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

        dm = HPatchesDataManager(raw_data_path=args.path, logger=logger)
        descriptor = Descriptor.create(descriptor_name=args.descriptor, logger=logger, config=config['descriptor'])
        preprocessor = Preprocessor(config=config['preprocessor'], logger=logger)

        full_dataset = dm.load_dataset(num_scenes=args.num_scenes)

        fake_kp = [cv.KeyPoint(32, 32, 31)]
        NORM_MAP = { cv.NORM_L2: 'L2', cv.NORM_L1: 'L1'}
        dist_name = NORM_MAP.get(descriptor.default_norm, 'L2')
        all_descriptors = {'distance': dist_name}

        logger.info("Computing descriptors for all scenes")
        for scene_name, scene_data in full_dataset.items():
            scene_descr = {}

            for img_type, patches_batch in scene_data.items():
                list_of_descr = []

                for patch in patches_batch:

                    features = {'kp': fake_kp, 'des': None}
                    features = descriptor.compute(patch, features)
                    features = preprocessor.prepare_features(from_algo=args.descriptor, to_algo='opencv', features=features)

                    if features['des'] is None:
                        pass

                    list_of_descr.append(features['des'])

                scene_descr[img_type] = np.vstack(list_of_descr)

            all_descriptors[scene_name] = scene_descr


        logger.info(f"Running {args.task} Task evaluation")
        task = HPatchesTask.create(args.task, logger)
        results = task.eval_task(all_descriptors, {'test': list(full_dataset.keys())})
        task.report_metrics(results, args.task)

        logger.info("Pipeline finished successfully")
        return 0

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main() or 0)

