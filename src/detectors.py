from __future__ import annotations
from logging import Logger
from typing import Any
import numpy as np
import cv2 as cv
from abc import ABC, abstractmethod


class Detector(ABC):
    _METHODS: dict[str, type[Detector]] = {}

    def __init__(self, logger: Logger, detector_name: str = 'sift') -> None:
        self._detector_name = detector_name
        self._logger = logger

    def __init_subclass__(cls, register: bool = True, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if register:
            key = cls.__name__.replace("Detector", "").lower()
            if key:
                Detector._METHODS[key] = cls

    @staticmethod
    def create(detector_name: str, logger: Logger, **kwargs: Any) -> Detector:
        if detector_name not in Detector._METHODS:
            raise ValueError(f"Detector '{detector_name}' not found."
                             f" Available: {list(Detector._METHODS.keys())}")

        return Detector._METHODS[detector_name](detector_name, logger, **kwargs)

    @abstractmethod
    def detect(self, img: np.ndarray) -> tuple[cv.KeyPoint, ...]:
        pass


class OpenCVDetector(Detector, register = False):
    def __init__(self, detector_name: str, logger: Logger, extractor: cv.Feature2D) -> None:
        super().__init__(logger, detector_name)
        self._extractor = extractor

    def detect(self, img: np.ndarray) -> tuple[cv.KeyPoint, ...]:
        if img is None:
            self._logger.error(f"Input image is None. Detection aborted.")
            return ()

        self._logger.info(f"Detecting keypoints with {self._detector_name}")
        kp = self._extractor.detect(img, None)

        if kp:
            self._logger.info(f"{self._detector_name} found {len(kp)} points")
        else:
            self._logger.warning(f"{self._detector_name} found 0 points")
        return kp


class SIFTDetector(OpenCVDetector):
    def __init__(self, detector_name: str, logger: Logger, **kwargs: Any) -> None:
        super().__init__(detector_name, logger, cv.SIFT_create(**kwargs))


class ORBDetector(OpenCVDetector):
    def __init__(self, detector_name: str, logger: Logger, **kwargs: Any) -> None:
        super().__init__(detector_name, logger, cv.ORB_create(**kwargs))


class FASTDetector(OpenCVDetector):
    def __init__(self, detector_name: str, logger: Logger, **kwargs: Any) -> None:
        super().__init__(detector_name, logger, cv.FastFeatureDetector_create(**kwargs))


class AKAZEDetector(OpenCVDetector):
    def __init__(self, detector_name: str, logger: Logger, **kwargs: Any) -> None:
        super().__init__(detector_name, logger, cv.AKAZE_create(**kwargs))


class BRISKDetector(OpenCVDetector):
    def __init__(self, detector_name: str, logger: Logger, **kwargs: Any) -> None:
        super().__init__(detector_name, logger, cv.BRISK_create(**kwargs))


class KAZEDetector(OpenCVDetector):
    def __init__(self, detector_name: str, logger: Logger, **kwargs: Any) -> None:
        super().__init__(detector_name, logger, cv.KAZE_create(**kwargs))





