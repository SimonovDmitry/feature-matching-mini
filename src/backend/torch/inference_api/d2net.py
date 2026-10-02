import sys
from pathlib import Path
import torch
import numpy as np
from src.backend.inference_api_base import InferenceAPI
from src.backend.torch.inference_api.inference_api import TorchInferenceAPI

D2_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "3rdparty" / "d2net"
if str(D2_ROOT) not in sys.path:
    sys.path.insert(0, str(D2_ROOT))

from lib.pyramid import process_multiscale


@InferenceAPI.register("d2net_torch")
class D2NetTorchInferenceApi(TorchInferenceAPI):
    def run(self, inputs):
        img = inputs.get("image")

        try:
            with torch.no_grad():
                keypoints, scores, descriptors = process_multiscale(
                    torch.from_numpy(img).float().unsqueeze(0).to(self._device),
                    self._model, scales=[1])

            keypoints = keypoints[:, [1, 0]].astype(np.float32)
            return {'kp': keypoints, 'des': descriptors, 'sc': scores}

        except Exception as e:
            self._logger.error(f"D2-Net inference error: {e}")
            return {'kp': (), 'des': ()}
