from abc import ABC, abstractmethod
import cv2 as cv


class Matcher(ABC):
    _METHODS = {}

    def __init__(self, logger, matcher_name, descriptor_name, **kwargs):
        self.matcher_name = matcher_name
        self.descriptor_name = descriptor_name
        self.logger = logger

    def __init_subclass__(cls, register=True, **kwargs):
        super().__init_subclass__(**kwargs)

        if register:
            key = cls.__name__.replace("Matcher", "").lower()
            if key:
                Matcher._METHODS[key] = cls

    @staticmethod
    def create(matcher_name, logger, descriptor_name, **kwargs):
        matcher_class_name = Matcher._METHODS.get(matcher_name.lower())
        if not matcher_class_name:
            raise ValueError(f"Matcher '{matcher_name}' not found."
                             f" Available: {list(Matcher._METHODS.keys())}")

        return Matcher._METHODS[matcher_name](logger, matcher_name, descriptor_name, **kwargs)

    @abstractmethod
    def match(self, **kwargs):
        pass

    @abstractmethod
    def _init_matcher(self):
        pass


class OpenCVMatcher(Matcher, register=False):
    def __init__(self, logger, matcher_name, descriptor_name,  mode, **kwargs):
        super().__init__(logger, matcher_name, descriptor_name)
        self.mode = mode

    def match(self, **kwargs):
        des1 = kwargs.get('des1')
        if des1 is None:
            raise ValueError("des1 is None")
        des2 = kwargs.get('des2')
        if des2 is None:
            raise ValueError("des2 is None")

        matcher = self._init_matcher()
        if (self.mode == 'simple'):
            return matcher.match(des1, des2)
        elif (self.mode == 'knn'):
            k = kwargs.get('k')
            if k is None:
                k = 2
            return matcher.knnMatch(des1, des2, k)
        else:
            raise ValueError(f"Mode '{self.mode}' is not supported.")


class BFMatcher(OpenCVMatcher):
    def __init__(self, logger, matcher_name, descriptor_name, **kwargs):
        mode = kwargs.get('mode')
        if mode is None:
            mode = 'simple'
        super().__init__(logger, matcher_name, descriptor_name, mode, **kwargs)

    def _init_matcher(self):
        return cv.BFMatcher(self.descriptor_name.default_norm)


class FLANNMatcher(OpenCVMatcher):
    def __init__(self, logger, matcher_name, descriptor_name, **kwargs):
        mode = kwargs.get('mode')
        if mode is None:
            mode = 'simple'
        super().__init__(logger, matcher_name, descriptor_name, mode, **kwargs)
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

    def _get_default_index_params(self):
        if self.descriptor_name.defaultNorm == cv.NORM_HAMMING:
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

    def _init_matcher(self):
        return cv.FlannBasedMatcher(self.index_params, self.search_params)