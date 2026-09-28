from src.backend.model_loader_base import ModelLoader
from src.backend.torch.utils import get_device


class TorchModelLoader(ModelLoader):
    def __init__(self, model_name, model_path=None, config=None, logger=None):
        super().__init__(model_name=model_name, model_path=model_path, config=config, logger=logger)

        device = config.get('device', None)
        if device is None:
            self._device = get_device()
        else:
            self._device = device

    def load(self):
        pass
