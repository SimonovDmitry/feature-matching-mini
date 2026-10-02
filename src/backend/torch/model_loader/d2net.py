import torch
from pathlib import Path
from src.backend.model_loader_base import ModelLoader
from src.backend.torch.model_loader.model_loader import TorchModelLoader
from src.backend.torch.model_loader.weights_url import WEIGHTS_URL_MODELS

from thirdparty.d2net.lib.model_test import D2Net as D2NetModel


@ModelLoader.register("d2net_torch")
class D2NetTorchModelLoader(TorchModelLoader):
    def load(self):
        checkpoint = Path(self._model_path or self._config.pop('checkpoint', "weights/d2net/d2_tf.pth"))
        weights_url = WEIGHTS_URL_MODELS.get(self._model_name)

        use_relu = self._config.get('use_relu', True)
        device_str = self._config.get('device', 'cpu').lower()
        use_cuda = 'cuda' in device_str

        if not checkpoint.exists():
            self._logger.info(f"Downloading D2Net weights to {checkpoint}")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(weights_url, str(checkpoint))

        self._logger.info(f"Initializing D2NetModel from {checkpoint}")
        self._model = D2NetModel(model_file=str(checkpoint), use_relu=use_relu, use_cuda=use_cuda)
        self._model = self._model.to(self._device)
        self._model.eval()
        return {'model': self._model}
