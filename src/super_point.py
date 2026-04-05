import torch
import cv2 as cv
import numpy as np
from transformers import AutoImageProcessor, SuperPointForKeypointDetection

from src.detectors import Detector
from src.descriptors import Descriptor


class SuperPoint(Detector, Descriptor):
    _shared_model = None
    _shared_processor = None
    _shared_cache = {}

    def __init__(self, extractor_name, logger, device=None, threshold=0.005):
        Detector.__init__(self, logger, extractor_name)
        Descriptor.__init__(self, logger, extractor_name)

        if device is None:
            if torch.cuda.is_available():
                self._device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self._device = torch.device('mps')
            else:
                self._device = torch.device('cpu')
        else:
            self._device = device

        self._threshold = threshold
        checkpoint = "weights/superpoint"

        if SuperPoint._shared_model is None:
            SuperPoint._shared_processor = AutoImageProcessor.from_pretrained(checkpoint, local_files_only=True)
            SuperPoint._shared_model = SuperPointForKeypointDetection.from_pretrained(checkpoint,
                                                                                      local_files_only=True).to(
                                                                                      self._device)
            SuperPoint._shared_model.eval()

        self._processor = SuperPoint._shared_processor
        self._model = SuperPoint._shared_model

    @property
    def default_norm(self):
        return cv.NORM_L2

    def _forward(self, img):
        if img is None:
            self._logger.error("Input image is None. Detection aborted.")
            return (), ()

        img_id = id(img)
        if img_id in SuperPoint._shared_cache:
            return SuperPoint._shared_cache[img_id]

        self._logger.info(f"Running inference with {self._detector_name}")

        if len(img.shape) == 2:
            img_input = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:
            img_input = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        else:
            img_input = img
        inputs = self._processor(img_input, return_tensors="pt").to(self._device)

        try:
            with torch.no_grad():
                outputs = self._model(**inputs)

            image_size = img.shape[:2]
            processed = self._processor.post_process_keypoint_detection(outputs, [image_size])[0]

            raw_kp = processed['keypoints'].cpu().numpy()
            raw_scores = processed['scores'].cpu().numpy()
            raw_des = processed['descriptors'].cpu().numpy().astype(np.float32)

            mask = raw_scores > self._threshold
            kp = [cv.KeyPoint(x=float(p[0]), y=float(p[1]), size=8, response=float(s))
                        for p, s in zip(raw_kp[mask], raw_scores[mask])]
            des = raw_des[mask]

            if len(SuperPoint._shared_cache) > 10:
                first_key = next(iter(SuperPoint._shared_cache))
                del SuperPoint._shared_cache[first_key]

            SuperPoint._shared_cache[img_id] = (kp, des)

            if kp:
                self._logger.info(f"{self._detector_name} found {len(kp)} points")
            else:
                self._logger.warning(f"{self._detector_name} found 0 points")

            if des is not None:
                self._logger.info(f"{self._descriptor_name} computed {len(des)} descriptors")
            else:
                self._logger.warning(f"{self._descriptor_name} computed 0 descriptors")

            return kp, des

        except Exception as e:
            self._logger.warning(f"{self._detector_name} inference failed (likely 0 points): {e}")
            return (), ()

    def detect(self, img):
        return {'kp': self._forward(img)[0]}

    def compute(self, img, features):
        kp, des = self._forward(img)
        return {'kp': kp, 'des': des}

    def detectAndCompute(self, img):
        kp, des = self._forward(img)
        return {'kp': kp, 'des': des}