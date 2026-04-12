import cv2 as cv
import torch

from src.descriptors import Descriptor
from src.detectors import Detector
from lightglue import SuperPoint, DISK, SIFT, ALIKED, DoGHardNet


class LightGlueFeatureExtractor(Detector, Descriptor, register=False):
    _EXTRACTOR_CLASSES = {
        'superpoint_lightglue': SuperPoint,
        'disk_lightglue': DISK,
        'sift_lightglue': SIFT,
        'aliked_lightglue': ALIKED,
        'doghardnet_lightglue': DoGHardNet
    }

    _shared_models = {}
    _is_extracted = False
    _extracted_data = {}

    def __init__(self, extractor_name, logger, config=None):
        Detector.__init__(self, logger, extractor_name)
        Descriptor.__init__(self, logger, extractor_name)

        if config is None:
            config = {}

        device = config.pop('device', None)
        if device is None:
            if torch.cuda.is_available():
                self._device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self._device = torch.device('mps')
            else:
                self._device = torch.device('cpu')
        else:
            self._device = torch.device(device)

        self._extractor_name = extractor_name.lower()

        model_key = (self._extractor_name, self._device.type)
        if model_key not in LightGlueFeatureExtractor._shared_models:
            extractor_class = self._EXTRACTOR_CLASSES.get(self._extractor_name)
            if not extractor_class:
                raise ValueError(f"Extractor '{extractor_name}' not found.")

            self._logger.info(f"Loading {self._extractor_name} weights onto {self._device}")
            LightGlueFeatureExtractor._shared_models[model_key] = (extractor_class(**config).eval().to(self._device))

        self._extractor = LightGlueFeatureExtractor._shared_models[model_key]

    @property
    def default_norm(self):
        return cv.NORM_L2

    def _forward(self, image_tensor):
        if image_tensor is None:
            self._logger.error("Input image tensor is None.")
            return {'kp': (), 'des': ()}

        with torch.no_grad():
            if image_tensor.ndim == 3:
                image_tensor = image_tensor[None]

            self._logger.info(f"Running inference with {self._extractor_name}")
            LightGlueFeatureExtractor._extracted_data = self._extractor.extract(image_tensor.to(self._device))

            self._logger.info(f"{self._extractor_name} found"
                              f" {LightGlueFeatureExtractor._extracted_data['keypoints'].shape[1]} points")
            return LightGlueFeatureExtractor._extracted_data

    def detect(self, image_tensor):
        LightGlueFeatureExtractor._is_extracted = True
        return self._forward(image_tensor)

    def compute(self, image_tensor, features=None):
        if LightGlueFeatureExtractor._is_extracted:
            LightGlueFeatureExtractor._is_extracted = False
            return LightGlueFeatureExtractor._extracted_data
        else:
            return self._forward(image_tensor)

    def detectAndCompute(self, img):
        return self._forward(img)
