import torch
import numpy as np
from src.backend.inference_api_base import InferenceAPI
from src.backend.torch.inference_api.inference_api import TorchInferenceAPI

from lib.pyramid import process_multiscale  # noqa: E402
from lib.utils import preprocess_image  # noqa: E402


@InferenceAPI.register("d2net_torch")
class D2NetTorchInferenceApi(TorchInferenceAPI):
    def _preprocess(self, img):
        if torch.is_tensor(img):
            img_np = img.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
            if img_np.max() <= 1.0:
                img_np = (img_np * 255)
        else:
            img_np = np.array(img)

        return preprocess_image(img_np, preprocessing='caffe')

    def run(self, inputs):
        img = inputs.get('image')

        try:
            with torch.no_grad():
                keypoints, scores, descriptors = process_multiscale(
                    torch.from_numpy(img).float().unsqueeze(0).to(self._device),
                    self._model, scales=[1])

            mask = scores > self._threshold
            kp = keypoints[mask]
            des = descriptors[mask]
            sc = scores[mask]

            if self._nfeatures is not None and len(kp) > self._nfeatures:
                top_indices = np.argsort(sc)[::-1][:self._nfeatures]

                keypoints = kp[top_indices]
                descriptors = des[top_indices]
                scores = sc[top_indices]

            if len(kp) > 0:
                keypoints = keypoints[:, [1, 0]].astype(np.float32)

                return {
                    'keypoints': keypoints,
                    'scores': scores,
                    'descriptors': descriptors
                }
            else:
                return {'keypoints': (), 'descriptors': ()}

        except Exception as e:
            self._logger.error(f"D2-Net inference error: {e}")
            return {'keypoints': (), 'descriptors': ()}
