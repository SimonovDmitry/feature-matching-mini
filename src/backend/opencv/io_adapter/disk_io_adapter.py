import cv2 as cv
import numpy as np
from src.backend.io_adapter_base import IOAdapter
from src.backend.opencv.io_adapter.io_adapter import OpenCVIOAdapter


@IOAdapter.register("disk_opencv")
class DiskOpenCVIOAdapter(OpenCVIOAdapter):
    def preprocess(self, inputs):
        img = inputs.pop('image')
        h, w = img.shape[:2]
        blob = cv.dnn.blobFromImage(img, scalefactor=1.0/255.0, size=(w, h), swapRB=True, crop=False)
        self._model.setInput(blob)

    def postprocess(self, outputs):
        keypoints = outputs[0]
        descriptors = np.squeeze(outputs[1], axis=0)
        scores = np.squeeze(outputs[2], axis=0)
        keypoints_np = keypoints.astype(np.float32)

        if keypoints_np.ndim == 3:
            keypoints_np = keypoints_np.reshape(-1, 2)

        keypoints = cv.KeyPoint_convert(keypoints_np)
        keypoints = np.asarray(keypoints)

        result = {'kp': keypoints, 'des': descriptors, 'sc': scores}
        return result
