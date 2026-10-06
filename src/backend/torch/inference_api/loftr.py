import torch

from src.backend.inference_api_base import InferenceAPI
from src.backend.torch.inference_api.inference_api import TorchInferenceAPI


@InferenceAPI.register("loftr_torch")
class LoFTRTorchInferenceApi(TorchInferenceAPI):
    def run(self, inputs):
        try:
            with torch.no_grad():
                self._model(inputs)

            return inputs

        except Exception as e:
            self._logger.error(f"D2-Net inference error: {e}")
            return {'kp': (), 'des': (), 'matches': ()}
