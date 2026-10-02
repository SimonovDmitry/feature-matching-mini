import sys
from pathlib import Path
import torch
from src.backend.inference_api_base import InferenceAPI
from src.backend.torch.inference_api.inference_api import TorchInferenceAPI

R2D2_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "3rdparty" / "r2d2"
if str(R2D2_ROOT) not in sys.path:
    sys.path.insert(0, str(R2D2_ROOT))

from extract import extract_multiscale


@InferenceAPI.register("r2d2_torch")
class R2D2TorchInferenceApi(TorchInferenceAPI):
    def run(self, inputs):
        img = inputs.get("image")
        model = inputs.get("model")
        nms = inputs.get("nms")

        try:
            with torch.no_grad():
                xys, desc, scores = extract_multiscale(model, img, nms)

            return {'kp': xys.cpu(),
                    'des': desc.cpu(),
                    'sc': scores.cpu()}

        except Exception as e:
            self._logger.warning(f"R2D2 inference failed (likely 0 points): {e}")
            return {'keypoints': (), 'descriptors': ()}


