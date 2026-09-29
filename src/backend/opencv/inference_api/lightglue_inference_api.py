import cv2 as cv
from src.backend.inference_api_base import InferenceAPI
from src.backend.opencv.inference_api.inference_api import OpenCVInferenceAPI

@InferenceAPI.register("lightglue_opencv")
class LightGlueOpenCVInferenceAPI(OpenCVInferenceAPI):
    def run(self, inputs):
        try:
            features1, features2 = inputs.get('features1'), inputs.get('features2')
            des1 = features1.get('des')
            des2 = features2.get('des')
            kp1 = features1.get('kp')
            kp2 = features2.get('kp')
            img_shape1 = features1.get('img_shape')
            img_shape2 = features2.get('img_shape')
            h1, w1 = img_shape1[:2]
            h2, w2 = img_shape2[:2]

            if not kp1 or not kp2 or des1 is None or des2 is None:
                return {'matches': ()}

            kpts1_mat = cv.KeyPoint_convert(kp1)
            kpts2_mat = cv.KeyPoint_convert(kp2)
            self._model.setPairInfo(kpts1_mat, kpts2_mat, (w1, h1), (w2, h2))
            if self._mode == 'knn':
                matches = self._model.knnMatch(des1, des2, k=1)
                valid_matches = [m for m in matches if m]
                self._logger.info(f"LightGlue found {len(valid_matches)} matches")
            else:
                matches = self._model.match(des1, des2)
                self._logger.info(f"LightGlue found {len(matches)} matches")
            return {'matches': matches}

        except Exception as e:
            self._logger.error(f"LightGlue inference error: {e}")
            return {'matches': ()}
