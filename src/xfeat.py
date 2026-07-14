import cv2 as cv
import torch
import sys
from pathlib import Path

XFEAT_ROOT = str(Path(__file__).parent.parent / "xfeat_repo")

if XFEAT_ROOT not in sys.path:
    sys.path.append(XFEAT_ROOT)

from src.detectors import Detector
from src.descriptors import Descriptor
from src.image_utils import to_numpy_bgr

from xfeat_repo.modules.xfeat import XFeat as XFeatModel


class XFeat(Detector, Descriptor):
    _model = None
    _extracted_data = {}
    _is_extracted = False

    def __init__(self, extractor_name, logger, config=None):
        if config is None:
            config = {}

        Detector.__init__(self, logger, extractor_name)
        Descriptor.__init__(self, logger, extractor_name)

        device = config.pop('device', None)
        self._threshold = config.pop('threshold', 0.005)
        self._top_k = config.pop('top_k', 4096)

        if config:
            self._logger.warning(f"XFeat: unknown config keys ignored: {list(config.keys())}")

        if device is None:
            if torch.cuda.is_available():
                self._device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self._device = torch.device('mps')
            else:
                self._device = torch.device('cpu')
        else:
            self._device = torch.device(device)

        if XFeat._model is None:
            try:
                self._logger.info(f"Loading XFeat weights onto {self._device}")
                XFeat._model = XFeatModel().to(self._device)
                XFeat._model.dev = self._device
                XFeat._model.eval()
            except Exception as e:
                self._logger.error(f"Failed to load XFeat: {e}")

        self._model = XFeat._model

    @property
    def default_norm(self):
        return cv.NORM_L2

    def _forward(self, img):
        if img is None:
            self._logger.error("Input image is None. Detection aborted.")
            return {'keypoints': (), 'descriptors': ()}

        self._logger.info(f"Running inference with {self._detector_name}")

        input_type = 'torch' if isinstance(img, torch.Tensor) else 'numpy'
        img_np = to_numpy_bgr(img, input_type=input_type)
        img_rgb = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)

        input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0)
        input_tensor = input_tensor.to(self._device) / 255.0

        try:
            with torch.no_grad():
                output = self._model.detectAndCompute(input_tensor, top_k=self._top_k)[0]

            raw_kp = output['keypoints'].cpu()
            raw_des = output['descriptors'].cpu()
            raw_scores = output['scores'].cpu()

            mask = raw_scores > self._threshold

            XFeat._extracted_data = {
                'keypoints': raw_kp[mask],
                'descriptors': raw_des[mask],
                'scores': raw_scores[mask].numpy()
            }

            if len(raw_kp[mask]) > 0:
                self._logger.info(f"{self._detector_name} found {len(raw_kp[mask])} points")
            else:
                self._logger.warning(f"{self._detector_name} found 0 points")

            if raw_des[mask] is not None:
                self._logger.info(f"{self._descriptor_name} computed {len(raw_des[mask])} descriptors")
            else:
                self._logger.warning(f"{self._descriptor_name} computed 0 descriptors")

            return XFeat._extracted_data

        except Exception as e:
            self._logger.warning(f"{self._detector_name} inference failed (likely 0 points): {e}")
            return {'keypoints': (), 'descriptors': (), 'scores': ()}

    def detect(self, img):
        XFeat._is_extracted = True
        return self._forward(img)

    def compute(self, img, kp):
        if XFeat._is_extracted:
            XFeat._is_extracted = False
            return XFeat._extracted_data
        else:
            return self._forward(img)

    def detectAndCompute(self, img):
        return self._forward(img)