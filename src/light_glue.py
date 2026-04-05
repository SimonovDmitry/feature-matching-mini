import torch

from src.descriptors import Descriptor
from src.detectors import Detector
from src.matchers import Matcher
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import rbd


class LightGlueExtractor(Detector, Descriptor):
    _shared_models = {}
    _shared_cache = {}

    _EXTRACTOR_CLASSES = {
        'superpoint': SuperPoint,
        'disk': DISK,
        'sift': SIFT,
        'aliked': ALIKED,
        'doghardnet': DoGHardNet
    }

    def __init__(self, logger, extractor_name, device=None, max_num_keypoints=2048):
        Detector.__init__(self, logger, extractor_name)
        Descriptor.__init__(self, logger, extractor_name)

        if device is None:
            if torch.cuda.is_available():
                self._device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self._device = torch.device('mps')
            else:
                self._device = torch.device('cpu')
        else:
            self._device = device

        self._extractor_name = extractor_name.lower()

        model_key = (self._extractor_name, self._device.type)
        if model_key not in LightGlueExtractor._shared_models:
            extractor_class = self._EXTRACTOR_CLASSES.get(self._extractor_name)
            if not extractor_class:
                raise ValueError(f"Extractor '{extractor_name}' not found.")

            self._logger.info(f"Loading {self._extractor_name} weights onto {self._device}")
            LightGlueExtractor._shared_models[model_key] = (extractor_class(max_num_keypoints=max_num_keypoints)
                                                            .eval().to(self._device))

        self._extractor = LightGlueExtractor._shared_models[model_key]

    def _forward(self, image_tensor):
        if image_tensor is None:
            self._logger.error("Input image tensor is None.")
            return {}

        img_id = id(image_tensor)
        if img_id in LightGlueExtractor._shared_cache:
            return LightGlueExtractor._shared_cache[img_id]

        with torch.no_grad():
            if image_tensor.ndim == 3:
                image_tensor = image_tensor[None]

            self._logger.info(f"Running inference with {self._extractor_name}")
            features = self._extractor.extract(image_tensor.to(self._device))

            if len(LightGlueExtractor._shared_cache) > 10:
                first_key = next(iter(LightGlueExtractor._shared_cache))
                del LightGlueExtractor._shared_cache[first_key]

            LightGlueExtractor._shared_cache[img_id] = features

            self._logger.info(f"{self.extractor_name} found {features['keypoints'].shape[1]} points")
            return features

    def detect(self, image_tensor):
        return self._forward(image_tensor)

    def compute(self, image_tensor, features=None):
        return self._forward(image_tensor)


class LightGlueMatcher(Matcher):
    _shared_matchers = {}

    def __init__(self, logger, extractor_type='superpoint', device='cpu'):
        Matcher.__init__(self, logger, "lightglue")
        self._device = torch.device(device)
        self.extractor_type = extractor_type.lower()

        matcher_key = (self.extractor_type, self._device.type)
        if matcher_key not in LightGlueMatcher._shared_matchers:
            self._logger.info(f"Loading LightGlue matcher for {self.extractor_type}")
            LightGlueMatcher._shared_matchers[matcher_key] = (LightGlue(features=self.extractor_type)
                                                              .eval().to(self._device))

        self._matcher = LightGlueMatcher._shared_matchers[matcher_key]

    @staticmethod
    def create(logger, extractor_type='superpoint', device='cuda'):
        return LightGlueMatcher(logger, extractor_type, device)

    def match(self, features0, features1):
        with torch.no_grad():
            input_dict = {"image0": features0, "image1": features1}

            matches01 = self._matcher(input_dict)
            matches01 = rbd(matches01)

            return matches01["matches"]