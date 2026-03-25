import cv2 as cv
from abc import ABC, abstractmethod


class Detector(ABC):
    _METHODS = {}

    def __init__(self, logger, detector_name = 'sift'):
        self._detector_name = detector_name
        self._logger = logger

    def __init_subclass__(cls, register = True, **kwargs):
        super().__init_subclass__(**kwargs)

        if register:
            key = cls.__name__.replace("Detector", "").lower()
            if key:
                Detector._METHODS[key] = cls

    @staticmethod
    def create(detector_name, logger, **kwargs):
        if detector_name not in Detector._METHODS:
            raise ValueError(f"Detector '{detector_name}' not found."
                             f" Available: {list(Detector._METHODS.keys())}")

        return Detector._METHODS[detector_name](detector_name, logger, **kwargs)

    @abstractmethod
    def detect(self, img):
        pass


class OpenCVDetector(Detector, register=False):
    def __init__(self, detector_name, logger, extractor):
        super().__init__(logger, detector_name)
        self._extractor = extractor

    def detect(self, img):
        if img is None:
            self._logger.error("Input image is None. Detection aborted.")
            return ()

        self._logger.info(f"Detecting keypoints with {self._detector_name}")
        kp = self._extractor.detect(img, None)

        if kp:
            self._logger.info(f"{self._detector_name} found {len(kp)} points")
        else:
            self._logger.warning(f"{self._detector_name} found 0 points")
        return kp


class SIFTDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, **kwargs):
        super().__init__(detector_name, logger, cv.SIFT_create(**kwargs))


class ORBDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, **kwargs):
        super().__init__(detector_name, logger, cv.ORB_create(**kwargs))


class FASTDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, **kwargs):
        super().__init__(detector_name, logger, cv.FastFeatureDetector_create(**kwargs))


class AKAZEDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, **kwargs):
        super().__init__(detector_name, logger, cv.AKAZE_create(**kwargs))


class BRISKDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, **kwargs):
        super().__init__(detector_name, logger, cv.BRISK_create(**kwargs))


class KAZEDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, **kwargs):
        super().__init__(detector_name, logger, cv.KAZE_create(**kwargs))


class GFTTDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, **kwargs):
        super().__init__(detector_name, logger, cv.GFTTDetector_create(**kwargs))


class MSERDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, **kwargs):
        super().__init__(detector_name, logger, cv.MSER_create(**kwargs))


class AGASTDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, **kwargs):
        super().__init__(detector_name, logger, cv.AgastFeatureDetector_create(**kwargs))


class BlobDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, **kwargs):
        super().__init__(detector_name, logger, cv.SimpleBlobDetector_create(**kwargs))


class StarDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, **kwargs):
        super().__init__(detector_name, logger, cv.xfeatures2d.StarDetector_create(**kwargs))


class HarrisLaplaceDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, **kwargs):
        super().__init__(detector_name, logger, cv.xfeatures2d.HarrisLaplaceFeatureDetector_create(**kwargs))


class MSDDetector(OpenCVDetector):
    def __init__(self, detector_name, logger, **kwargs):
        super().__init__(detector_name, logger, cv.xfeatures2d.MSDDetector_create(**kwargs))
