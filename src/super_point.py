import torch
import cv2 as cv
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
from pathlib import Path

from src.detectors import Detector
from src.descriptors import Descriptor
from src.image_utils import to_numpy_bgr


class SuperPoint(Detector, Descriptor):
    _model = None
    _image_processor = None

    _is_extracted = False
    _extracted_data = {}

    def __init__(self, extractor_name, logger, config=None):
        if config is None:
            config = {}

        Detector.__init__(self, logger, extractor_name)
        Descriptor.__init__(self, logger, extractor_name)

        device = config.pop('device', None)
        self._threshold = config.pop('threshold', 0.005)
        checkpoint = config.pop('checkpoint', "weights/superpoint")
        local_files_only = config.pop('local_files_only', True)

        if config:
            self._logger.warning(f"SuperPoint: unknown config keys ignored: {list(config.keys())}")
        remote_repo = "magic-leap-community/superpoint"

        if device is None:
            if torch.cuda.is_available():
                self._device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self._device = torch.device('mps')
            else:
                self._device = torch.device('cpu')
        else:
            self._device = torch.device(device)

        if SuperPoint._model is None:
            local_path = Path(checkpoint)
            if not local_path.exists() or not any(local_path.iterdir()):
                self._logger.warning(f"Local checkpoint {checkpoint} not found or empty.")
                self._logger.info(f"Switching to remote repository: {remote_repo}")
                checkpoint = remote_repo
                local_files_only = False

            try:
                self._logger.info(f"Loading SuperPoint from {checkpoint} (local={local_files_only})")
                SuperPoint._image_processor = AutoImageProcessor.from_pretrained(
                    checkpoint, local_files_only=local_files_only)
                SuperPoint._model = SuperPointForKeypointDetection.from_pretrained(
                    checkpoint, local_files_only=local_files_only).to(self._device)
            except Exception as e:
                self._logger.error(f"Failed to load from {checkpoint}: {e}")

        self._processor = SuperPoint._image_processor
        self._model = SuperPoint._model

    @property
    def default_norm(self):
        return cv.NORM_L2

    def _forward(self, img):
        if img is None:
            self._logger.error("Input image is None. Detection aborted.")
            return {'keypoints': (), 'descriptors': ()}

        self._logger.info(f"Running inference with {self._detector_name}")

        input_type = 'torch' if isinstance(img, torch.Tensor) else 'numpy'
        img = to_numpy_bgr(img, input_type=input_type)
        inputs = self._processor(img, return_tensors="pt").to(self._device)

        try:
            with torch.no_grad():
                outputs = self._model(**inputs)

            processed = self._processor.post_process_keypoint_detection(outputs, [img.shape[:2]])[0]

            raw_kp = processed['keypoints']
            raw_scores = processed['scores']
            raw_des = processed['descriptors']

            mask = raw_scores > self._threshold
            SuperPoint._extracted_data = {
                'keypoints': raw_kp[mask],
                'descriptors': raw_des[mask],
                'scores': raw_scores[mask]
            }

            if len(raw_kp[mask]) > 0:
                self._logger.info(f"{self._detector_name} found {len(raw_kp[mask])} points")
            else:
                self._logger.warning(f"{self._detector_name} found 0 points")

            if raw_des[mask] is not None:
                self._logger.info(f"{self._descriptor_name} computed {len(raw_des[mask])} descriptors")
            else:
                self._logger.warning(f"{self._descriptor_name} computed 0 descriptors")

            return SuperPoint._extracted_data

        except Exception as e:
            self._logger.warning(f"{self._detector_name} inference failed (likely 0 points): {e}")
            return {'keypoints': (), 'descriptors': ()}

    def detect(self, img):
        SuperPoint._is_extracted = True
        return self._forward(img)

    def compute(self, img, features):
        if SuperPoint._is_extracted:
            SuperPoint._is_extracted = False
            return SuperPoint._extracted_data
        else:
            return self._forward(img)

    def detectAndCompute(self, img):
        return self._forward(img)
