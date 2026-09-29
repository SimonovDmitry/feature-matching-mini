import cv2 as cv
import numpy as np

from src.descriptors import Descriptor
from src.detectors import Detector
from src.backend.inference_api_base import InferenceAPI
from src.backend.model_loader_base import ModelLoader


class OpenCVDNNFeatureExtractors(Detector, Descriptor, register=False):
    _is_extracted = False
    _extracted_data = {}

    def __init__(self, extractor_name, logger, config):
        Detector.__init__(self, logger, extractor_name)
        Descriptor.__init__(self, logger, extractor_name)
        self.extractor_name = extractor_name

        loader_name = f"{extractor_name.lower()}_opencv"

        self._loader = ModelLoader.create(backend=loader_name, model_name=extractor_name, config=config, logger=logger)
        self._model = self._loader.load()
        self._inference = InferenceAPI.create(backend=loader_name, logger=logger, model_name=extractor_name,
                                              model=self._model, config=config)

    @property
    def default_norm(self):
        return cv.NORM_L2

    def _forward(self, img):
        if img is None:
            self._logger.error("Input image is None. Detection aborted.")
            return {'kp': (), 'des': ()}

        self._logger.info(f"Running inference with {self._detector_name}")

        output = self._inference.run(img)
        kp = output.get('keypoints', np.array([]))
        des = output.get('descriptors', np.array([]))
        self._logger.info(f"{self.extractor_name} found {len(kp)} points")

        OpenCVDNNFeatureExtractors._extracted_data = {'kp': kp, 'des': des, 'img_shape': img.shape}
        return OpenCVDNNFeatureExtractors._extracted_data

    def detect(self, img):
        OpenCVDNNFeatureExtractors._is_extracted = True
        return self._forward(img)

    def compute(self, img, features):
        if OpenCVDNNFeatureExtractors._is_extracted:
            OpenCVDNNFeatureExtractors._is_extracted = False
            return OpenCVDNNFeatureExtractors._extracted_data
        else:
            return self._forward(img)

    def detectAndCompute(self, img):
        return self._forward(img)


class ALIKEDOpenCV(OpenCVDNNFeatureExtractors):
    def __init__(self, extractor_name, logger, config):
        super().__init__(extractor_name, logger, config)


class DISKOpenCV(OpenCVDNNFeatureExtractors):
    def __init__(self, extractor_name, logger, config):
        super().__init__(extractor_name, logger, config)
