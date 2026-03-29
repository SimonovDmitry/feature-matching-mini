import torch
import cv2 as cv
import numpy as np
from transformers import AutoImageProcessor, SuperPointForKeypointDetection

from src.detectors import Detector
from src.descriptors import Descriptor


class SuperPoint(Detector, Descriptor):
    _shared_model = None
    _shared_processor = None

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

        self._kp = None
        self._des = None
        self._last_img_id = None

    @property
    def default_norm(self):
        return cv.NORM_L2

    def _forward(self, img):
        if img is None:
            self._logger.error("Input image is None. Detection aborted.")
            return (), ()

        if id(img) == self._last_img_id:
            return self._kp, self._des

        self._logger.info(f"Running inference with {self._detector_name}")
        img_input = cv.cvtColor(img, cv.COLOR_BGR2RGB) if len(img.shape) == 3 else img
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
            self._kp = [cv.KeyPoint(x=float(p[0]), y=float(p[1]), size=8, response=float(s))
                        for p, s in zip(raw_kp[mask], raw_scores[mask])]
            self._des = raw_des[mask]
            self._last_img_id = id(img)

        except Exception as e:
            self._logger.warning(f"{self._detector_name} inference failed (likely 0 points): {e}")
            self._kp = []
            self._des = np.zeros((0, 256), dtype=np.float32)
            self._last_img_id = None

        if self._kp:
            self._logger.info(f"{self._detector_name} found {len(self._kp)} points")
        else:
            self._logger.warning(f"{self._detector_name} found 0 points")

        if self._des is not None:
            self._logger.info(f"{self._descriptor_name} computed {len(self._des)} descriptors")
        else:
            self._logger.warning(f"{self._descriptor_name} computed 0 descriptors")

        return self._kp, self._des

    def detect(self, img):
        return self._forward(img)[0]

    def compute(self, img, kp):
        model_kp, model_des = self._forward(img)

        if len(kp) == len(model_kp):
            return kp, model_des

        self._logger.warning("SuperPoint compute() called with external KeyPoints. Returning internal descriptors.")
        return model_kp, model_des

    def detectAndCompute(self, img):
        return self._forward(img)