import cv2 as cv
import numpy as np

from src.detectors import Detector
from src.descriptors import Descriptor
import src.backend.torch.model_loader
import src.backend.torch.inference_api
import src.backend.torch.io_adapter
from src.backend.io_adapter_base import IOAdapter
from src.backend.inference_api_base import InferenceAPI
from src.backend.model_loader_base import ModelLoader


class DNNFeatureExtractors(Detector, Descriptor):
    _model = None
    _is_extracted = False
    _extracted_data = {}

    def __init__(self, extractor_name, logger, config=None):
        if config is None:
            config = {}

        Detector.__init__(self, logger, extractor_name)
        Descriptor.__init__(self, logger, extractor_name)

        backend = config.pop('backend', 'torch').lower()
        loader_name = f"{extractor_name.lower()}_{backend}"

        self._loader = ModelLoader.create(backend=loader_name, model_name=extractor_name, config=config, logger=logger)
        self._model = self._loader.load()
        self._io_adapter = IOAdapter.create(backend=loader_name, model_name=extractor_name, config=config,
                                            logger=logger)
        self._inference = InferenceAPI.create(backend=loader_name, logger=logger, model_name=extractor_name,
                                              model=self._model, config=config)

    def _forward(self, img):
        if img is None:
            self._logger.error("Input image is None. Detection aborted.")
            return {'keypoints': (), 'descriptors': ()}

        self._logger.info(f"Running inference with {self._detector_name}")
        inputs = {'image': img}

        inputs = self._io_adapter.preprocess(inputs)
        outputs = self._inference.run(inputs)
        outputs = self._io_adapter.postprocess(outputs)

        keypoints = outputs.get('kp', np.array([]))
        descriptors = outputs.get('des', np.array([]))
        scores = outputs.get('sc', np.array([]))

        extracted = {'kp': keypoints,
                     'des': descriptors,
                     'sc': scores}
        DNNFeatureExtractors._extracted_data = extracted

        if len(keypoints) > 0:
            self._logger.info(f"{self._detector_name} found {len(keypoints)} points")
        else:
            self._logger.warning(f"{self._detector_name} found 0 points")

        if descriptors is not None:
            self._logger.info(f"{self._descriptor_name} computed {len(descriptors)} descriptors")
        else:
            self._logger.warning(f"{self._descriptor_name} computed 0 descriptors")

        return extracted

    @property
    def default_norm(self):
        return cv.NORM_L2

    def detect(self, img):
        DNNFeatureExtractors._is_extracted = True
        return self._forward(img)

    def compute(self, img, features=None):
        if DNNFeatureExtractors._is_extracted:
            DNNFeatureExtractors._is_extracted = False
            return self._extracted_data
        else:
            return self._forward(img)

    def detectAndCompute(self, img):
        return self._forward(img)
