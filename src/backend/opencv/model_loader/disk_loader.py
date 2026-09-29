import cv2 as cv
from src.backend.model_loader_base import ModelLoader
from src.backend.opencv.model_loader.model_loader import OpenCVModelLoader

@ModelLoader.register("disk_opencv")
class DiskOpenCVModelLoader(OpenCVModelLoader):
    _PARAM_MAPPING = {
        'nfeatures': 'maxKeypoints',
        'threshold': 'scoreThreshold'
    }

    def load(self):
        checkpoint = self._config.pop('disk_model_path', "models/disk_1024.onnx")
        self._logger.info(f"Initializing Disk from {checkpoint}")

        mapped_config = {}
        for config_key, value in self._config.items():
            if config_key in self._PARAM_MAPPING:
                mapped_config[self._PARAM_MAPPING[config_key]] = value

        self._model = cv.DISK.create(checkpoint, **mapped_config)
        return self._model
