import sys
from pathlib import Path
from src.backend.model_loader_base import ModelLoader
from src.backend.torch.model_loader.model_loader import TorchModelLoader

R2D2_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "3rdparty" / "r2d2"
if str(R2D2_ROOT) not in sys.path:
    sys.path.insert(0, str(R2D2_ROOT))

from extract import NonMaxSuppression, load_network


@ModelLoader.register("r2d2_torch")
class R2D2TorchModelLoader(TorchModelLoader):
    def load(self):
        checkpoint = Path(self._model_path or self._config.pop('checkpoint', "3rdparty/r2d2/models/r2d2_WASF_N16.pt"))

        self._logger.info(f"Loading R2D2 weights from {checkpoint}")
        self._model = load_network(checkpoint).to(self._device).eval()
        self._nms = NonMaxSuppression()
        return {'model': self._model, 'nms': self._nms}
