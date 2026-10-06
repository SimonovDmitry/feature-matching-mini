import torch
from src.backend.inference_api_base import InferenceAPI
from src.backend.torch.inference_api.inference_api import TorchInferenceAPI


@InferenceAPI.register("xfeat_torch")
class XFeatTorchInferenceApi(TorchInferenceAPI):
    def run(self, inputs):
        img = inputs.get("image")

        try:
            with torch.no_grad():
                output = self._model.detectAndCompute(img, top_k=self._nfeatures)[0]

            return {'keypoints': output['keypoints'],
                    'descriptors': output['descriptors'],
                    'scores': output['scores']}

        except Exception as e:
            self._logger.warning(f"XFeat inference error: {e}")
            return {'keypoints': (), 'descriptors': (), 'scores': ()}
