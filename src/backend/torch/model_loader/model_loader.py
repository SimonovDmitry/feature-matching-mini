from src.backend.model_loader_base import ModelLoader
from src.backend.torch.utils import get_device

from lib.model_test import D2Net as D2NetModel  # noqa: E402
from lib.pyramid import process_multiscale  # noqa: E402
from lib.utils import preprocess_image  # noqa: E402


class TorchModelLoader(ModelLoader):
    def __init__(self, model_name, model, model_path=None, config=None, logger=None):
        super().__init__(model_name=model_name, model_path=model_path, config=config, logger=logger)

        device = config.get('device', None)
        if device is None:
            self._device = get_device()
        else:
            self._device = device
        self._model = model

    def load(self):
        pass
