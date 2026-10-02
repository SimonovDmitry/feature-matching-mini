import torch
from src.backend.inference_api_base import InferenceAPI
from src.backend.torch.inference_api.inference_api import TorchInferenceAPI


@InferenceAPI.register("superpoint_torch")
class SuperPointTorchInferenceApi(TorchInferenceAPI):
    def run(self, inputs):
        img = inputs.get("image")
        processor = inputs.get("processor")
        height = inputs.get("height")
        width = inputs.get("width")

        try:
            with torch.no_grad():
                outputs = self._model(**img)

            processed = processor.post_process_keypoint_detection(outputs, [[height, width]])[0]
            raw_kp = processed['keypoints']
            raw_scores = processed['scores']
            raw_des = processed['descriptors']

            return {'kp': raw_kp,
                    'des': raw_des,
                    'sc': raw_scores,
                    'width': width,
                    'height': height}

        except Exception as e:
            self._logger.error(f"Super Point inference error: {e}")
            return {'kp': (), 'des': ()}
