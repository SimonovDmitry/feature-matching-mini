import cv2 as cv
from src.matchers import Matcher
from src.backend.inference_api_base import InferenceAPI
from src.backend.model_loader_base import ModelLoader

class LightGlueOpenCVMatcher(Matcher):
    def __init__(self, matcher_name, logger, config, descriptor_name):
        super().__init__(matcher_name, logger, config, descriptor_name)

        if descriptor_name._descriptor_name == 'aliked' and config.get('lightglue_model_path') is None:
            config['lightglue_model_path'] = "models/lightglue_for_aliked.onnx"

        loader_name = f"{matcher_name.lower()}_opencv"

        self._loader = ModelLoader.create(backend=loader_name, model_name=matcher_name, config=config, logger=logger)
        self._model = self._loader.load()
        self._inference = InferenceAPI.create(backend=loader_name, logger=logger, model_name=matcher_name,
                                              model=self._model, config=config)

    def match(self, features1, features2):
        output = self._inference.run({'features1': features1, 'features2': features2})
        return {'matches': output['matches']}
