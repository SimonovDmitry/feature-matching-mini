import torch
from src.backend.inference_api_base import InferenceAPI
from src.backend.torch.inference_api.inference_api import TorchInferenceAPI


@InferenceAPI.register("superglue_torch")
class SuperGlueTorchInferenceApi(TorchInferenceAPI):
    def run(self, inputs):
        try:
            with torch.no_grad():
                outputs = self._model(inputs)

            matches = outputs['matches0']
            scores = outputs['matching_scores0']
            return {'matches' : matches, 'scores' : scores, **inputs}

        except Exception as e:
            self._logger.error(f"Super Glue inference error: {e}")
            return {'matches': (), 'scores': ()}

