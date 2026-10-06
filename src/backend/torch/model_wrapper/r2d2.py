import sys
from pathlib import Path
import torch
from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.model_loader import TorchModelWrapper

R2D2_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "3rdparty" / "r2d2"
if str(R2D2_ROOT) not in sys.path:
    sys.path.insert(0, str(R2D2_ROOT))

from extract import NonMaxSuppression, load_network
from extract import extract_multiscale


@ModelWrapper.register("r2d2_torch")
class R2D2TorchModelWrapper(TorchModelWrapper):
    def load(self):
        checkpoint = Path(self._model_path or self._config.pop('checkpoint', "3rdparty/r2d2/models/r2d2_WASF_N16.pt"))

        self._logger.info(f"Loading R2D2 weights from {checkpoint}")
        self._model = load_network(checkpoint).to(self._device).eval()
        self._nms = NonMaxSuppression()
        return {'model': self._model, 'nms': self._nms}

    def call(self, inputs):
        img = inputs.get("image")
        model = inputs.get("model")
        nms = inputs.get("nms")

        try:
            with torch.no_grad():
                xys, desc, scores = extract_multiscale(model, img, nms)

            return {'keypoints': xys.cpu(),
                    'descriptors': desc.cpu(),
                    'scores': scores.cpu()}

        except Exception as e:
            self._logger.warning(f"R2D2 inference failed (likely 0 points): {e}")
            return {'keypoints': (), 'descriptors': ()}
