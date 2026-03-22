from __future__ import annotations
from logging import Logger
from typing import Any
import numpy as np
import cv2 as cv
from abc import ABC, abstractmethod


class Descriptor(ABC):
    _METHODS: dict[str, type[Descriptor]] = {}

    def __init__(self, logger: Logger, descriptor_name: str = 'sift') -> None:
        self._descriptor_name = descriptor_name
        self._logger = logger

    def __init_subclass__(cls, register: bool = True, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if register:
            key = cls.__name__.replace("Descriptor", "").lower()
            if key:
                Descriptor._METHODS[key] = cls

    @staticmethod
    def create(descriptor_name: str, logger: Logger, **kwargs: Any) -> Descriptor:
        if descriptor_name not in Descriptor._METHODS:
            raise ValueError(f"Descriptor '{descriptor_name}' not found."
                             f" Available: {list(Descriptor._METHODS.keys())}")

        return Descriptor._METHODS[descriptor_name](descriptor_name, logger, **kwargs)

    @property
    @abstractmethod
    def default_norm(self) -> int:
        pass

    @abstractmethod
    def compute(self, img: np.ndarray, kp: tuple[cv.KeyPoint, ...]) \
            -> tuple[tuple[cv.KeyPoint, ...], np.ndarray | None]:
        pass


class OpenCVDescriptor(Descriptor, register=False):
    def __init__(self, descriptor_name: str, logger: Logger, extractor: cv.Feature2D) -> None:
        super().__init__(logger, descriptor_name)
        self._extractor = extractor

    @property
    def default_norm(self) -> int:
        return self._extractor.defaultNorm()

    def compute(self, img: np.ndarray, kp: tuple[cv.KeyPoint, ...]) \
            -> tuple[tuple[cv.KeyPoint, ...], np.ndarray | None]:
        if img is None:
            self._logger.error(f"Input image is None. Detection aborted.")
            return ()

        self._logger.info(f"Computing {self._descriptor_name} descriptors")
        kp, des = self._extractor.compute(img, kp)

        if des is not None:
            self._logger.info(f"{self._descriptor_name} computed {len(des)} descriptors")
        else:
            self._logger.warning(f"{self._descriptor_name} computed 0 descriptors")
        return kp, des


class SIFTDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name: str, logger: Logger, **kwargs) -> None:
        super().__init__(descriptor_name, logger, cv.SIFT_create(**kwargs))


class ORBDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name: str, logger: Logger, **kwargs) -> None:
        super().__init__(descriptor_name, logger, cv.ORB_create(**kwargs))


class AKAZEDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name: str, logger: Logger, **kwargs) -> None:
        super().__init__(descriptor_name, logger, cv.AKAZE_create(**kwargs))


class BRISKDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name: str, logger: Logger, **kwargs: Any) -> None:
        super().__init__(descriptor_name, logger, cv.BRISK_create(**kwargs))


class KAZEDescriptor(OpenCVDescriptor):
    def __init__(self, descriptor_name: str, logger: Logger, **kwargs: Any) -> None:
        super().__init__(descriptor_name, logger, cv.KAZE_create(**kwargs))
