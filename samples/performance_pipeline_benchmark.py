import sys
import logging
import numpy as np

from src.image_utils import read_image
from samples.utils import build_config, performance_tests_parser
from src.preprocessor import Preprocessor
from src.algorithms import DNN_ALGORITHMS
from src.detectors import Detector
from src.descriptors import Descriptor
from src.matchers import Matcher
from samples.performance_profiler import PerformanceProfiler


logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger("Pipeline_Performance_Test_Sample")


def pipeline_performance_test(logger, profiler, detector, detector_name, descriptor, descriptor_name,
                              matcher, matcher_name, img0, img1, iterations):
    times = []

    if detector_name in DNN_ALGORITHMS or descriptor_name in DNN_ALGORITHMS:
        for _ in range(iterations):
            res, time = profiler.profile_dnn_pipeline(descriptor, descriptor_name, matcher, matcher_name, img0, img1)
            times.append(time)
    else:
        for _ in range(iterations):
            res, time = profiler.profile_pipeline(detector, detector_name, descriptor, descriptor_name,
                                                  matcher, matcher_name, img0, img1)
            times.append(time)
    min_time = np.min(times)
    mean_time = np.mean(times)
    logger.info(f"\nMin time pipeline test: {min_time}\n"
                f"Mean time pipeline test: {mean_time}")


def main():
    args = performance_tests_parser()

    try:
        if not args.image1.exists() or not args.image2.exists():
            logger.error(f"One of the images does not exist: {args.image1} or {args.image2}")
            return 1

        config = build_config(args)
        logger.info(f"Config: {config}")

        detector_config = config.get('detector', {})
        descriptor_config = config.get('descriptor', {})
        matcher_config = config.get('matcher', {})
        preprocessor_config = config.get('preprocessor', {})

        img0 = read_image(args.image1)
        img1 = read_image(args.image2)

        detector = Detector.create(detector_name=args.detector, logger=logger, config=detector_config)
        descriptor = Descriptor.create(descriptor_name=args.descriptor, logger=logger,
                                       config=descriptor_config)
        matcher = Matcher.create(matcher_name=args.matcher, descriptor_name=descriptor,
                                 logger=logger, config=matcher_config)
        preprocessor = Preprocessor(config=preprocessor_config, logger=logger)

        img0 = preprocessor.prepare_image(img0, from_algo='opencv', to_algo=args.detector)
        img1 = preprocessor.prepare_image(img1, from_algo='opencv', to_algo=args.detector)

        profiler = PerformanceProfiler(preprocessor)

        iterations = args.iterations
        logger.info(f"Running pipeline performance test with {iterations} iterations...")

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
