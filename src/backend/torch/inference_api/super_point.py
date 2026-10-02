import torch
from src.backend.inference_api_base import InferenceAPI
from src.backend.torch.inference_api.inference_api import TorchInferenceAPI


@InferenceAPI.register("superpoint_torch")
class SuperPointTorchInferenceApi(TorchInferenceAPI):
    def run(self, inputs):
        img = inputs.get("image")

        try:
            with torch.no_grad():
                outputs = self._model(**img)

            return {'inference_outputs' : outputs, **inputs}

        except Exception as e:
            self._logger.error(f"Super Point inference error: {e}")
            return {'kp': (), 'des': ()}
