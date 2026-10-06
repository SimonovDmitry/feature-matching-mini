import sys
from pathlib import Path
import torch

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.torch.model_wrapper.model_loader import TorchModelWrapper

ROMA_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / '3rdparty' / 'roma'
if str(ROMA_ROOT) not in sys.path:
    sys.path.insert(0, str(ROMA_ROOT))

from romatch import roma_outdoor  # noqa: E402


@ModelWrapper.register("roma_torch")
class RoMaTorchModelWrapper(TorchModelWrapper):
    def load(self):
        coarse_res = self._config.get('coarse_res', 560)
        upsample_res = self._config.get('upsample_res', (864, 1152))

        self._logger.info(f"Loading RoMa weights onto {self._device}")
        self._model = roma_outdoor(device=self._device, coarse_res=coarse_res, upsample_res=upsample_res)
        self._model.eval().to(self._device)
        return {'model': self._model}

    def call(self, inputs):
        img0 = inputs.get("image0")
        img1 = inputs.get("image1")
        height0, width0 = inputs.get("height0"), inputs.get("width0")
        height1, width1 = inputs.get("height1"), inputs.get("width1")

        try:
            with torch.no_grad():
                warp, certainty = self._model.match(img0, img1, device=self._device)
                matches, conf = self._model.sample(warp, certainty, num=self._nfeatures)

                kp0, kp1 = self._model.to_pixel_coordinates(matches, height0, width0, height1, width1)

            return {'keypoints0': kp0.cpu(), 'keypoints1': kp1.cpu(), 'scores': conf.cpu()}

        except Exception as e:
                self._logger.error(f"RoMa match error: {e}")
                return {'matches': (), 'scores': ()}
