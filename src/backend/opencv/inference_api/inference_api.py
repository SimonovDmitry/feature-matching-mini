from src.backend.inference_api_base import InferenceAPI

class OpenCVInferenceAPI(InferenceAPI):
    def __init__(self, logger, model_name, model, config=None):
        if config is None:
            config = {}

        self._mode = config.get('mode', 'simple')
        super().__init__(logger, model_name, model, config)

    def run(self, inputs):
        pass
