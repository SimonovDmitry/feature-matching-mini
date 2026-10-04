import sys
from pathlib import Path

SG_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "3rdparty" / "superglue"
if str(SG_ROOT) not in sys.path:
    sys.path.insert(0, str(SG_ROOT))

from models.superglue import SuperGlue
from src.backend.model_loader_base import ModelLoader
from src.backend.torch.model_loader.model_loader import TorchModelLoader


@ModelLoader.register("superglue_torch")
class SuperGlueTorchModelLoader(TorchModelLoader):
    def load(self):
        sg_config = {
            'weights': self._config.get('weights', 'outdoor'),
            'sinkhorn_iterations': self._config.get('sinkhorn_iterations', 20),
            'match_threshold': self._config.get('threshold', 0.005),
        }

        self._logger.info(f"Loading SuperGlue ({sg_config.get('weights')}) onto {self._device}")
        self._model = SuperGlue(sg_config).to(self._device).eval()
        return {"model": self._model}
