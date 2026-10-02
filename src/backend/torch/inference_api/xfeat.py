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

            raw_kp = output['keypoints'].cpu()
            raw_des = output['descriptors'].cpu()
            raw_scores = output['scores'].cpu()

            return {'kp': raw_kp,
                    'des': raw_des,
                    'sc': raw_scores}

        except Exception as e:
            self._logger.warning(f"XFeat inference failed (likely 0 points): {e}")
            return {'kp': (), 'des': (), 'sc': ()}
