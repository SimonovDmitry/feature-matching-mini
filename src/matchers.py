from __future__ import annotations
from abc import ABC, abstractmethod
import cv2 as cv
from cv2 import DMatch
from numpy import ndarray
from logging import Logger
from typing import Any, List, Dict
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

    def __init__(self, logger: Logger, descriptor_method: Descriptor, mode: str = 'simple', **kwargs: Any) -> None:
        self._descriptor_method = descriptor_method
        self._mode = self._MODES.get(mode.lower())
        if not self._mode:
            raise ValueError(f"Mode '{mode}' not found ")
        self.logger = logger

    @staticmethod
    def create(matcher_name: str, logger: Logger, descriptor_method: Descriptor, mode: str, **kwargs: Any) -> Matcher:
        matcher_class_name = Matcher._METHODS.get(matcher_name.lower())
        if not matcher_class_name:
            raise ValueError(f"Matcher '{matcher_name}' not found ")
        matcher_class = globals()[matcher_class_name]
        return matcher_class(logger, descriptor_method, mode, **kwargs)

    def match(self, des1: ndarray, des2: ndarray, k: int = 2) -> Any:
        mode_match = getattr(self, self._mode)
        if self._mode == '_knn_match':
            return mode_match(des1, des2, k)
        else:
            return mode_match(des1, des2)

    @abstractmethod
    def _init_matcher(self):
        pass

    @abstractmethod
    def _simple_match(self, des1, des2):
        pass

    @abstractmethod
    def _knn_match(self, des1, des2, k):
        pass


class BFMatcher(Matcher):
    def __init__(self, logger: Logger, descriptor_method: Descriptor, mode: str = 'simple', **kwargs: Any) -> None:
        super().__init__(logger, descriptor_method, mode, **kwargs)

    def _init_matcher(self) -> cv.BFMatcher:
        return cv.BFMatcher(self._descriptor_method.default_norm)

    def _simple_match(self, des1: ndarray, des2: ndarray) -> List[DMatch]:
        bf = self._init_matcher()
        return bf.match(des1, des2)

    def _knn_match(self, des1: ndarray, des2: ndarray, k: int = 2) -> List[List[DMatch]]:
        bf = self._init_matcher()
        return bf.knnMatch(des1, des2, k)


class FLANNMatcher(Matcher):
    def __init__(self, logger: Logger, descriptor_method: Descriptor, mode: str = 'simple', **kwargs: Any) -> None:
        super().__init__(logger, descriptor_method, mode, **kwargs)
        self.index_params = kwargs.get('index_params')
        self.search_params = kwargs.get('search_params')

        if self.index_params is None:
            self.index_params = self._get_default_index_params()
        elif not isinstance(self.index_params, dict):
            raise TypeError("index_params must be dict")
        if self.search_params is None:
            self.search_params = {'checks': 50}
        elif not isinstance(self.search_params, dict):
            raise TypeError("search_params must be dict")

    def _get_default_index_params(self) -> Dict:
        if self._descriptor_method.defaultNorm == cv.NORM_HAMMING:
            return {
                'algorithm': 6,
                'table_number': 6,
                'key_size': 12,
                'multi_probe_level': 1
            }
        else:
            return {
                'algorithm': 1,
                'trees': 5
            }

    def _init_matcher(self) -> cv.FlannBasedMatcher:
        return cv.FlannBasedMatcher(self.index_params, self.search_params)

    def _simple_match(self, des1: ndarray, des2: ndarray) -> List[DMatch]:
            flann = self._init_matcher()
            return flann.match(des1, des2)

    def _knn_match(self, des1: ndarray, des2: ndarray, k: int = 2) -> List[List[DMatch]]:
            flann = self._init_matcher()
            return flann.knnMatch(des1, des2, k)