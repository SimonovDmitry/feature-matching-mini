import time
from src.preprocessor import Preprocessor
from src.detectors import Detector
from src.descriptors import Descriptor
from src.matchers import Matcher
from src.super_point import SuperPoint
from src.lightglue_matcher import LightGlue
from src.lightglue_pipeline import LightGlueFeatureExtractor

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        work_time = end - start
        return result, work_time
    return wrapper

class PerformanceProfiler:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    @measure_time
    def profile_detection(self, detector, img):
        kp = detector.detect(img)
        return {'kp': kp}

    @measure_time
    def profile_descriptor(self, descriptor, img, kp):
        des = descriptor.compute(img, {'kp': kp})
        return {'des': des}

    @measure_time
    def profile_matching(self, matcher, des1, des2):
        matches = matcher.match({'des': des1}, {'des': des2})
        return {'matches': matches}

    @measure_time
    def profile_pipeline(self, detector, detector_name, descriptor, descriptor_name,
                         matcher, matcher_name, img0, img1):
        kp0 = detector.detect(img0)
        kp0 = self.preprocessor.prepare_features(kp0, from_algo=detector_name, to_algo=descriptor_name)
        kp1 = detector.detect(img1)
        kp1 = self.preprocessor.prepare_features(kp1, from_algo=detector_name, to_algo=descriptor_name)

        des0 = descriptor.compute(img0, kp0)['des']
        des0 = self.preprocessor.prepare_features(des0, from_algo=descriptor_name, to_algo=matcher_name)
        des1 = descriptor.compute(img1, kp1)['des']
        des1 = self.preprocessor.prepare_features(des1, from_algo=descriptor_name, to_algo=matcher_name)

        matches = matcher.match({'des': des0}, {'des': des1})

    @measure_time
    def profile_dnn_extractor(self, extractor, img):
        features = extractor.detectAndCompute(img)
        return {'features': features}

    @measure_time
    def profile_dnn_pipeline(self, extractor, extractor_name,
                             matcher, matcher_name, img0, img1):
        features0 = extractor.detectAndCompute(img0)
        features1 = extractor.detectAndCompute(img1)
        features0 = self.preprocessor.prepare_features(features0, from_algo=extractor_name, to_algo=matcher_name)
        features1 = self.preprocessor.prepare_features(features1, from_algo=extractor_name, to_algo=matcher_name)
        matches = matcher.match({'features0': features0}, {'features1': features1})
