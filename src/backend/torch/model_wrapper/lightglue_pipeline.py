import sys
from pathlib import Path
import torch
from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.model_loader import TorchModelWrapper

LIGHTGLUE_ROOT = Path(__file__).resolve().parent.parent / "3rdparty" / "lightglue"
if str(LIGHTGLUE_ROOT) not in sys.path:
    sys.path.insert(0, str(LIGHTGLUE_ROOT))

from lightglue import SuperPoint, DISK, SIFT, ALIKED, DoGHardNet  # noqa: E402


@ModelWrapper.register("light_glue_feature_extractor_torch")
class LightGlueFeatureExtractorTorchModelLoader(TorchModelWrapper):
    _EXTRACTOR_CLASSES = {
        'superpoint_lightglue_torch': SuperPoint,
        'disk_lightglue_torch': DISK,
        'sift_lightglue_torch': SIFT,
        'aliked_lightglue_torch': ALIKED,
        'doghardnet_lightglue_torch': DoGHardNet
    }
    _shared_models = {}

    def load(self):
        model_key = (self._model_name, self._device.type)
        if model_key not in LightGlueFeatureExtractorTorchModelLoader._shared_models:
            extractor_class = self._EXTRACTOR_CLASSES.get(f"{self._model_name}_torch")
            if not extractor_class:
                raise ValueError(f"Extractor '{self._model_name}' not found.")

            self._logger.info(f"Loading {self._model_name} weights onto {self._device}")
            LightGlueFeatureExtractorTorchModelLoader._shared_models[model_key] = \
                (extractor_class(**self._config).eval().to(self._device))

        self._model = LightGlueFeatureExtractorTorchModelLoader._shared_models[model_key]
        return {'model': self._model}

    def call(self, inputs):
        img = inputs.get("image")
        model = inputs.get("model")

        try:
            with torch.no_grad():
                if img.ndim == 3:
                    image_tensor = img[None]

                self._logger.info(f"Running inference with {self._model_name}")
                extracted = model.extract(image_tensor.to(self._device))
            return extracted

        except Exception as e:
            self._logger.error(f"D2-Net inference error: {e}")
            return {'keypoints': (), 'descriptors': ()}
