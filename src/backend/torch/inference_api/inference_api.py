from src.backend.inference_api_base import InferenceAPI
from src.backend.torch.utils import get_device


class TorchInferenceAPI(InferenceAPI):
    def __init__(self, logger, model_name, model, config=None):
        if config is None:
            config = {}

        super().__init__(logger, model_name, model, config)

        device = config.get('device', None)
        if device is None:
            self._device = get_device()
        else:
            self._device = device

        self._nfeatures = config.get('nfeatures', 4096)
        self._threshold = config.get('threshold', 0.005)
        self._model = model.to(self._device).eval()

    def run(self, inputs):
        pass
