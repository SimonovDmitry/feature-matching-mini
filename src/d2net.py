import torch
import numpy as np
import cv2 as cv
import sys
from pathlib import Path

from src.detectors import Detector
from src.descriptors import Descriptor

D2_ROOT = Path(__file__).resolve().parent.parent / "3rdparty" / "d2net"
if str(D2_ROOT) not in sys.path:
    sys.path.insert(0, str(D2_ROOT))

from lib.model_test import D2Net as D2NetModel  # noqa: E402
from lib.pyramid import process_multiscale  # noqa: E402
from lib.utils import preprocess_image  # noqa: E402


class D2Net(Detector, Descriptor):
    _model = None
    _is_extracted = False
    _extracted_data = {}

    def __init__(self, extractor_name, logger, config=None):
        if config is None:
            config = {}

        Detector.__init__(self, logger, extractor_name)
        Descriptor.__init__(self, logger, extractor_name)

        self._threshold = config.pop('threshold', 0.005)
        device = config.pop('device', None)
        if device is None:
            if torch.cuda.is_available():
                self._device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self._device = torch.device('mps')
            else:
                self._device = torch.device('cpu')
        else:
            self._device = torch.device(device)

        checkpoint = config.pop('checkpoint', "weights/d2net/d2_tf.pth")
        self._use_relu = config.pop('use_relu', True)
        if config:
            self._logger.warning(f"D2Net: unknown config keys ignored: {list(config.keys())}")

        if D2Net._model is None:
            self._logger.info(f"Loading D2Net weights onto {self._device}")
            use_cuda = self._device.type == 'cuda'
            D2Net._model = D2NetModel(model_file=checkpoint, use_relu=self._use_relu, use_cuda=use_cuda)
            D2Net._model.eval()

        self._model = D2Net._model

    @property
    def default_norm(self):
        return cv.NORM_L2

    def _forward(self, img):
        if img is None:
            self._logger.error("Input image is None. Detection aborted.")
            return {'keypoints': (), 'descriptors': ()}

        self._logger.info(f"Running inference with {self._detector_name}")

        if torch.is_tensor(img):
            img_np = img.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
            if img_np.max() <= 1.0:
                img_np = (img_np * 255)
        else:
            img_np = np.array(img)

        img_prep = preprocess_image(img_np, preprocessing='caffe')

        try:
            with torch.no_grad():
                keypoints, scores, descriptors = process_multiscale(
                    torch.from_numpy(img_prep).float().unsqueeze(0).to(self._device),
                    self._model, scales=[1])

            mask = scores > self._threshold
            raw_kp = keypoints[mask]

            if len(raw_kp) > 0:
                xy_coords = raw_kp[:, [1, 0]].astype(np.float32)

                extracted = {
                    'keypoints': torch.from_numpy(xy_coords),
                    'descriptors': torch.from_numpy(descriptors[mask]),
                    'scores': torch.from_numpy(scores[mask])
                }
                D2Net._extracted_data = extracted

                if len(keypoints[mask]) > 0:
                    self._logger.info(f"{self._detector_name} found {len(keypoints[mask])} points")
                else:
                    self._logger.warning(f"{self._detector_name} found 0 points")

                if descriptors[mask] is not None:
                    self._logger.info(f"{self._descriptor_name} computed {len(descriptors[mask])} descriptors")
                else:
                    self._logger.warning(f"{self._descriptor_name} computed 0 descriptors")

                return extracted
            else:
                return {'keypoints': (), 'descriptors': ()}

        except Exception as e:
            self._logger.error(f"D2-Net inference error: {e}")
            return {'keypoints': (), 'descriptors': ()}

    def detect(self, img):
        D2Net._is_extracted = True
        return self._forward(img)

    def compute(self, img, kp):
        if D2Net._is_extracted:
            D2Net._is_extracted = False
            return D2Net._extracted_data
        else:
            return self._forward(img)

    def detectAndCompute(self, img):
        return self._forward(img)
