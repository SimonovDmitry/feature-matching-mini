import cv2 as cv
from abc import ABC, abstractmethod


class Descriptor(ABC):
    _METHODS = {}

    def __init__(self, logger, descriptor_name='sift'):
        self._descriptor_name = descriptor_name
        self._logger = logger

    def __init_subclass__(cls, register=True, **kwargs):
        super().__init_subclass__(**kwargs)

        if register:
            key = cls.__name__.replace("Descriptor", "").lower()
            if key:
                Descriptor._METHODS[key] = cls

    @staticmethod
    def create(descriptor_name, logger, **kwargs):
        if descriptor_name not in Descriptor._METHODS:
            raise ValueError(f"Descriptor '{descriptor_name}' not found."
                             f" Available: {list(Descriptor._METHODS.keys())}")

        return Descriptor._METHODS[descriptor_name](descriptor_name, logger, **kwargs)

    @property
    @abstractmethod
    def default_norm(self):
        pass

    @abstractmethod
    def compute(self, img, kp):
        pass


class OpenCVDescriptor(Descriptor, register=False):
    def __init__(self, descriptor_name, logger, extractor):
        super().__init__(logger, descriptor_name)
        self._extractor = extractor

    @property
    def default_norm(self):
        return self._extractor.defaultNorm()

    def compute(self, img, features):
        if img is None:
            self._logger.error("Input image is None. Detection aborted.")
            return {'kp': (), 'des': ()}

        self._logger.info(f"Computing {self._descriptor_name} descriptors")
        kp, des = self._extractor.compute(img, features.get('kp'))

        if des is not None:
            self._logger.info(f"{self._descriptor_name} computed {len(des)} descriptors")
        else:
            self._logger.warning(f"{self._descriptor_name} computed 0 descriptors")
        return {'kp': kp, 'des': des}


class SIFTDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, **kwargs):
        super().__init__(descriptor_name, logger, cv.SIFT_create(**kwargs))


class ORBDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, **kwargs):
        super().__init__(descriptor_name, logger, cv.ORB_create(**kwargs))


class AKAZEDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, **kwargs):
        super().__init__(descriptor_name, logger, cv.AKAZE_create(**kwargs))


class BRISKDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, **kwargs):
        super().__init__(descriptor_name, logger, cv.BRISK_create(**kwargs))


class KAZEDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, **kwargs):
        super().__init__(descriptor_name, logger, cv.KAZE_create(**kwargs))


class BRIEFDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, **kwargs):
        super().__init__(descriptor_name, logger, cv.xfeatures2d.BriefDescriptorExtractor_create(**kwargs))


class FREAKDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, **kwargs):
        super().__init__(descriptor_name, logger, cv.xfeatures2d.FREAK_create(**kwargs))


class DAISYDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, **kwargs):
        super().__init__(descriptor_name, logger, cv.xfeatures2d.DAISY_create(**kwargs))


class LATCHDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, **kwargs):
        super().__init__(descriptor_name, logger, cv.xfeatures2d.LATCH_create(**kwargs))


class BEBLIDDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, **kwargs):
        super().__init__(descriptor_name, logger, cv.xfeatures2d.BEBLID_create(scale_factor=0.75, **kwargs))


class TEBLIDDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, **kwargs):
        super().__init__(descriptor_name, logger, cv.xfeatures2d.TEBLID_create(scale_factor=0.75, **kwargs))


class VGGDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, **kwargs):
        super().__init__(descriptor_name, logger, cv.xfeatures2d.VGG_create(**kwargs))


class BoostDescDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name, logger, **kwargs):
        super().__init__(descriptor_name, logger, cv.xfeatures2d.BoostDesc_create(**kwargs))
