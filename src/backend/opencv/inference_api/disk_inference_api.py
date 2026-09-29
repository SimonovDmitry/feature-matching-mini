from src.backend.inference_api_base import InferenceAPI
from src.backend.opencv.inference_api.inference_api import OpenCVInferenceAPI

@InferenceAPI.register("disk_opencv")
class DiskOpenCVInferenceAPI(OpenCVInferenceAPI):
    def run(self, img):
        try:
            kp, des = self._model.detectAndCompute(img, None)
            return {'kp': kp, 'des': des, 'img_shape': img.shape}

        except Exception as e:
            self._logger.error(f"Disk inference error: {e}")
            return {'keypoints': (), 'descriptors': ()}
