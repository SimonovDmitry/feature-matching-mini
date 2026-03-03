from abc import ABC, abstractmethod
import cv2 as cv


class BaseDescriptors(ABC):
    def __init__(self):
        self.method = None

    @abstractmethod
    def create_method(self):
        pass

    def detect_and_compute(self, img1, img2):
        kp1, des1 = self.method.detectAndCompute(img1, None)
        kp2, des2 = self.method.detectAndCompute(img2, None)
        return kp1, des1, kp2, des2


class SIFT(BaseDescriptors):
    def __init__(self):
        super().__init__()
        self.method = self.create_method()

    def create_method(self):
        return cv.SIFT_create()


class ORB(BaseDescriptors):
    def __init__(self):
        super().__init__()
        self.method = self.create_method()

    def create_method(self):
        return cv.ORB_create()


class Descriptors:
    _descriptors = {
        'sift': SIFT,
        'orb': ORB
    }

    def select_descriptors(self, method):
        method_class = self._descriptors.get(method.lower())
        if not method_class:
            raise ValueError(f"Method '{method}' not found ")
        return method_class()

    def apply_method(self, img1, img2, method):
        method_instance = self.select_descriptors(method)
        kp1, des1, kp2, des2 = method_instance.detect_and_compute(img1, img2)
        return kp1, des1, kp2, des2
