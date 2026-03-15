from __future__ import annotations
from abc import ABC, abstractmethod
import cv2 as cv
from cv2 import DMatch
from numpy import ndarray
from logging import Logger
from typing import Any, List
from src.descriptors import Descriptor

class Matcher(ABC):
    _METHODS = {
        'bf': 'BFMatcher',
        'flann': 'FLANNMatcher'
    }

    _MODES = {
        'knn': '_knn_match',
        'simple': '_simple_match'
    }

    def __init__(self, logger: Logger, descriptor_method: Descriptor, mode: str = 'simple') -> None:
        self._descriptor_method = descriptor_method
        self._mode = self._MODES.get(mode.lower())
        if not self._mode:
            raise ValueError(f"Mode '{mode}' not found ")
        self.logger = logger

    @staticmethod
    def create(matcher_name: str, logger: Logger, descriptor_method: Descriptor, mode: str) -> Matcher:
        matcher_class_name = Matcher._METHODS.get(matcher_name.lower())
        if not matcher_class_name:
            raise ValueError(f"Matcher '{matcher_name}' not found ")
        matcher_class = globals()[matcher_class_name]
        return matcher_class(logger, descriptor_method, mode)

    def match(self, des1: ndarray, des2: ndarray) -> Any:
        mode_match = getattr(self, self._mode)
        return mode_match(des1, des2)

    @abstractmethod
    def _init_matcher(self):
        pass

    @abstractmethod
    def _simple_match(self, des1, des2):
        pass

    @abstractmethod
    def _knn_match(self, des1, des2):
        pass


class BFMatcher(Matcher):
    def __init__(self, logger: Logger, descriptor_method: Descriptor, mode: str = 'simple') -> None:
        super().__init__(logger, descriptor_method, mode)

    def _init_matcher(self) -> cv.BFMatcher:
        return cv.BFMatcher(self._descriptor_method.default_norm)

    def _simple_match(self, des1: ndarray, des2: ndarray) -> List[DMatch]:
        bf = self._init_matcher()
        return bf.match(des1, des2)

    def _knn_match(self, des1: ndarray, des2: ndarray) -> List[List[DMatch]]:
        bf = self._init_matcher()
        return bf.knnMatch(des1, des2, k=2)


class FLANNMatcher(Matcher):
    def __init__(self, logger: Logger, descriptor_method: Descriptor, mode: str = 'simple') -> None:
        super().__init__(logger, descriptor_method, mode)

    def _init_matcher(self) -> cv.FlannBasedMatcher:
        if self._descriptor_method.default_norm == cv.NORM_HAMMING:
            flann_index_lsh = 6
            index_params = dict(algorithm=flann_index_lsh,
                                table_number=6,
                                key_size=12,
                                multi_probe_level=1)
        else:
            flann_index_kdtree = 1
            index_params = dict(algorithm=flann_index_kdtree, trees=5)
        search_params = dict(checks=50)
        return cv.FlannBasedMatcher(index_params, search_params)

    def _simple_match(self, des1: ndarray, des2: ndarray) -> List[DMatch]:
        flann = self._init_matcher()
        return flann.match(des1, des2)

    def _knn_match(self, des1: ndarray, des2: ndarray) -> List[List[DMatch]]:
        flann = self._init_matcher()
        return flann.knnMatch(des1, des2, k=2)