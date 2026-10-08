import sys
from pathlib import Path
import torch

LIGHTGLUE_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "3rdparty" / "lightglue"
if str(LIGHTGLUE_ROOT) not in sys.path:
    sys.path.insert(0, str(LIGHTGLUE_ROOT))

from lightglue import LightGlue
from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.model_loader import TorchModelWrapper


@ModelWrapper.register("lightglue_torch")
class LightGlueTorchModelLoader(TorchModelWrapper):
    def load(self):
        extractor_name = self._config.get('descriptor_name').replace("_lightglue", "")
        self._logger.info(f"Loading SuperGlue onto {self._device}")
        self._model = LightGlue(features=extractor_name, **self._config).eval().to(self._device)
        return {"model": self._model}

    def call(self, inputs):
        try:
            with torch.no_grad():
                outputs = self._model(inputs)
            return {"matches": outputs["matches"], "scores": outputs["scores"]}

        except Exception as e:
            self._logger.error(f"LightGlue inference error: {e}")
            return {"matches": (), "scores": ()}



