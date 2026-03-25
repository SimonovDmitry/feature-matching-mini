import cv2 as cv

from src.detectors import Detector
from src.descriptors import Descriptor
from src.matchers import Matcher


class FeatureMatcherCV2:
    _DETECTOR_DESCRIPTOR_COMPATIBILITY = {
        'sift': ['sift', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
        'orb': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
        'fast': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
        'akaze': ['sift', 'orb', 'akaze', 'brisk', 'kaze', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid',
                  'vgg', 'boostdesc'],
        'brisk': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
        'kaze': ['sift', 'orb', 'brisk', 'kaze', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg',
                 'boostdesc'],
        'gftt': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
        'mser': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
        'agast': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
        'blob': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
        'star': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
        'harrislaplace': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg',
                          'boostdesc'],
        'msd': ['orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc']
    }

    def __init__(self, logger, detector='sift', descriptor='sift',
                 matcher='bf', matcher_mode='simple'):
        self._detector = detector
        self._descriptor = descriptor
        self._matcher = matcher
        self._matcher_mode = matcher_mode
        self._logger = logger

        self._validate_compatibility()

    def _validate_compatibility(self):
        if self._detector not in self._DETECTOR_DESCRIPTOR_COMPATIBILITY:
            raise ValueError(f"Detector '{self._detector}' is not registered in compatibility matrix")

        if self._descriptor not in self._DETECTOR_DESCRIPTOR_COMPATIBILITY[self._detector]:
            raise ValueError(f"Detector {self._detector} cannot be used with Descriptor {self._descriptor}")

    def visualize_matches(self, img1, kp1, img2, kp2, matches):
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),
                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        if not matches or len(matches) == 0:
            self._logger.warning("No matches found to visualize.")
            return cv.drawMatches(img1, kp1, img2, kp2, [], None, **draw_params)

        if self._matcher_mode == 'simple':
            return cv.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
        if self._matcher_mode == 'knn':
            return cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

        return cv.drawMatches(img1, kp1, img2, kp2, [], None, **draw_params)

    def match(self, img1, img2):
        detector = Detector.create(detector_name=self._detector, logger=self._logger)
        kp1 = detector.detect(img1)
        kp2 = detector.detect(img2)

        if not kp1 or not kp2:
            raise ValueError("Failed to detect key points")

        descriptor = Descriptor.create(descriptor_name=self._descriptor, logger=self._logger)
        kp1, des1 = descriptor.compute(img1, kp1)
        kp2, des2 = descriptor.compute(img2, kp2)

        if des1 is None or des2 is None:
            raise ValueError("Descriptors could not be computed")

        matcher = Matcher.create(matcher_name=self._matcher, descriptor_method=descriptor,
                                 mode=self._matcher_mode, logger=self._logger)
        matches = matcher.match(des1, des2)
        return kp1, kp2, matches
