import cv2 as cv
from src.backend.model_loader_base import ModelLoader
from src.backend.opencv.model_loader.model_loader import OpenCVModelLoader

@ModelLoader.register("aliked_opencv")
class AlikedOpenCVModelLoader(OpenCVModelLoader):
    _PARAM_MAPPING = {
        'nfeatures': 'max_num_keypoints',
        'threshold': 'detection_threshold',
        'scale_factor': 'nms_radius'
    }

    def load(self):
        checkpoint = self._config.pop('aliked_model_path', "models/aliked-n32-top2k-640.onnx")
        self._logger.info(f"Initializing Aliked from {checkpoint}")

        mapped_config = {}
        for config_key, value in self._config.items():
            if config_key in self._PARAM_MAPPING:
                mapped_config[self._PARAM_MAPPING[config_key]] = value

        aliked_params = cv.ALIKED.Params()
        for key, value in mapped_config.items():
            if hasattr(aliked_params, key):
                setattr(aliked_params, key, value)

        self._model = cv.ALIKED.create(checkpoint, aliked_params)
        return self._model
