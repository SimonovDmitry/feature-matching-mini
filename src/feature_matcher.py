from logging import Logger
import numpy as np
import cv2 as cv
from cv2 import DMatch
from typing import Any

from src.detectors import Detector
from src.descriptors import Descriptor
from src.matchers import Matcher


class FeatureMatcherCV2:
    def __init__(self, logger: Logger, detector: str = 'sift', descriptor: str = 'sift',
                 matcher: str = 'bf', matcher_mode: str = 'sample') -> None:
        self._detector = detector
        self._descriptor = descriptor
        self._matcher = matcher
        self._matcher_mode = matcher_mode
        self._logger = logger

    def visualize_matches(self, img1: np.ndarray, kp1: tuple[cv.KeyPoint, ...], img2: np.ndarray,
                          kp2: tuple[cv.KeyPoint, ...], matches: Any) -> np.ndarray:
        if not matches or len(matches) == 0:
            self._logger.warning("No matches found to visualize.")
            return np.hstack((img1, img2))

        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),
                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        if self._matcher_mode == 'simple':
            return cv.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)

        if self._matcher_mode == 'knn':
            return cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

        return np.hstack((img1, img2))

    def match(self, img1: np.ndarray, img2: np.ndarray) \
            -> tuple[tuple[cv.KeyPoint, ...], tuple[cv.KeyPoint, ...], list[DMatch] | list[list[DMatch]]]:
        detector = Detector.create(detector_name=self._detector, logger=self._logger)
        kp1 = detector.detect(img1)
        kp2 = detector.detect(img2)

        if kp1 is None or kp2 is None:
            self._logger.warning("Failed to detect key points")
            return kp1 or (), kp2 or (), []

        descriptor = Descriptor.create(descriptor_name=self._descriptor, logger=self._logger)
        kp1, des1 = descriptor.compute(img1, kp1)
        kp2, des2 = descriptor.compute(img2, kp2)

        if des1 is None or des2 is None:
            self._logger.error("Descriptors could not be computed")
            return kp1, kp2, []

        matcher = Matcher.create(matcher_name=self._matcher, descriptor_method=descriptor,
                                 mode=self._matcher_mode, logger=self._logger)
        matches = matcher.match(des1, des2)

        return kp1, kp2, matches
