import sys
from pathlib import Path
from src.backend.model_loader_base import ModelLoader
from src.backend.torch.model_loader.model_loader import TorchModelLoader

XFEAT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "3rdparty" / "xfeat"
if str(XFEAT_ROOT) not in sys.path:
    sys.path.insert(0, str(XFEAT_ROOT))

from modules.xfeat import XFeat as XFeatModel


@ModelLoader.register("xfeat_torch")
class XFeatTorchModelLoader(TorchModelLoader):
    def load(self):
        self._logger.info(f"Loading XFeat weights onto {self._device}")
        self._model = XFeatModel().to(self._device)
        self._model.dev = self._device
        self._model.eval()
        return self._model
