import cv2 as cv
from src.backend.model_loader_base import ModelLoader
from src.backend.opencv.model_loader.model_loader import OpenCVModelLoader


@ModelLoader.register("lightglue_opencv")
class LightGlueOpenCVModelLoader(OpenCVModelLoader):
    def load(self):
        checkpoint = self._config.pop('lightglue_model_path', "models/disk_lightglue_2outputs.onnx")
        self._logger.info(f"Initializing LightGlue from {checkpoint}")
        self.scoreThreshold = self._config.pop('score_threshold', 0.1)

        self._model = cv.LightGlueMatcher.create(self.lightglue_model_path, scoreThreshold=self.scoreThreshold)
        return self._model
