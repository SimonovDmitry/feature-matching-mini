import cv2 as cv

from src.detectors import Detector
from src.descriptors import Descriptor
from src.super_point import SuperPoint
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
        'msd': ['orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
        'superpoint': ['superpoint']
    }

    def __init__(self, logger, detector='sift', descriptor='sift',
                 matcher='bf', matcher_mode=None):
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

    def visualize_matches(self, img1, features1, img2, features2, matches):
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),
                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        if not matches or len(matches) == 0:
            self._logger.warning("No matches found to visualize.")
            return cv.drawMatches(img1, features1.get('kp'), img2, features2.get('kp'),
                                  [], None, **draw_params)

        if self._matcher_mode == 'simple':
            return cv.drawMatches(img1, features1.get('kp'), img2, features2.get('kp'),
                                  matches, None, **draw_params)
        if self._matcher_mode == 'knn':
            return cv.drawMatchesKnn(img1, features1.get('kp'), img2, features2.get('kp'),
                                     matches, None, **draw_params)

        return cv.drawMatches(img1, features1.get('kp'), img2, features2.get('kp'),
                              [], None, **draw_params)

    def match(self, img1, img2):
        detector = Detector.create(detector_name=self._detector, logger=self._logger)
        features0 = detector.detect(img1)
        features1 = detector.detect(img2)

        if not features0.get('kp') or not features1.get('kp'):
            raise ValueError("Failed to detect key points")

        descriptor = Descriptor.create(descriptor_name=self._descriptor, logger=self._logger)
        features0 = descriptor.compute(img1, features0)
        features1 = descriptor.compute(img2, features1)

        if features0.get('des') is None or features1.get('des') is None:
            raise ValueError("Descriptors could not be computed")

        matcher = Matcher.create(matcher_name=self._matcher, descriptor_name=descriptor,
                                 mode=self._matcher_mode, logger=self._logger)
        matches = matcher.match(features0, features1)
        return features0, features1, matches
