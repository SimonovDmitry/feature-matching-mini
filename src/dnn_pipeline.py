from abc import abstractmethod
import numpy as np

from src.dnn_extractors import DNNFeatureExtractors
from src.dnn_matchers import DNNMatcher
import src.backend.torch.model_loader
import src.backend.torch.inference_api
import src.backend.torch.io_adapter
from src.backend.io_adapter_base import IOAdapter
from src.backend.inference_api_base import InferenceAPI
from src.backend.model_loader_base import ModelLoader


class DNNPipeline(DNNFeatureExtractors, DNNMatcher):
    def __init__(self, extractor_name, logger, config=None):
        if config is None:
            config = {}

        DNNFeatureExtractors.__init__(self, extractor_name, logger, config)
        DNNMatcher.__init__(self, extractor_name, logger, config, extractor_name)

        backend = config.pop('backend', 'torch').lower()
        loader_name = f"{extractor_name.lower()}_{backend}"

        self._nfeatures = config.get('nfeatures', 4096)
        self._threshold = config.get('threshold', 0.005)

        self._loader = ModelLoader.create(backend=loader_name, model_name=extractor_name, config=config, logger=logger)
        self._model_components = self._loader.load()
        self._model = self._model_components.get('model')
        self._io_adapter = IOAdapter.create(backend=loader_name, model_name=extractor_name, config=config,
                                            logger=logger)
        self._inference = InferenceAPI.create(backend=loader_name, logger=logger, model_name=extractor_name,
                                              model=self._model, config=config)
    def detect(self, img):
        return {'image': img, 'keypoints': (), 'descriptors': ()}

    def compute(self, img, features=None):
        return features

    def match(self, features0, features1):
        img0 = features0.get('image')
        img1 = features1.get('image')

        if img0 is None or img1 is None:
            self._logger.error("Input image is None")
            return {'matches': (), 'sc': ()}

        inputs = {'image0': img0, 'image1': img1}
        inputs = self._io_adapter.preprocess(inputs)
        outputs = self._inference.run(inputs)
        outputs = self._io_adapter.postprocess(outputs)

        matches = outputs.get('matches')
        scores = outputs.get('sc')
        keypoints0 = outputs.get('keypoints0')
        keypoints1 = outputs.get('keypoints1')

        if matches is None or scores is None:
            self._logger.warning(f"{self._detector_name}: invalid matcher output")
            return {'matches': (), 'sc': ()}

        mask = scores > self._threshold
        matches = matches[mask]
        scores = scores[mask]

        if self._nfeatures is not None and len(matches) > self._nfeatures:
            indices = np.argsort(scores)[::-1][:self._nfeatures]
            matches = matches[indices]
            scores = scores[indices]

        extracted = {'keypoints0': keypoints0,
                     'keypoints1': keypoints1,
                     'matches': matches,
                     'sc': scores}
        DNNPipeline._extracted_data = extracted

        if len(matches) > 0:
            self._logger.info(f"{self._detector_name} found {len(matches)} matches")
        else:
            self._logger.warning(f"{self._detector_name} found 0 matches")

        return extracted
