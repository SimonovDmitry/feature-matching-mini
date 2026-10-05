import sys
from pathlib import Path
import torch

from src.backend.model_loader_base import ModelLoader
from src.backend.torch.model_loader.model_loader import TorchModelLoader
from src.backend.torch.model_loader.weights_url import WEIGHTS_URL_MODELS

LOFTR_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "3rdparty" / "loftr" / "src"

if str(LOFTR_ROOT) not in sys.path:
    sys.path.insert(0, str(LOFTR_ROOT))

from loftr import LoFTR as LoFTRModel  # noqa: E402
from loftr import default_cfg  # noqa: E402


@ModelLoader.register("loftr_torch")
class LoFTRTorchModelLoader(TorchModelLoader):
    def load(self):
        self._weights_type = self._config.pop('weights', 'outdoor')
        checkpoint = Path(self._model_path or
                          self._config.pop('checkpoint', "weights/loftr/loftr_{self._weights_type}.ckpt"))
        weights_url = WEIGHTS_URL_MODELS.get(self._model_name).get(self._weights_type)

        if not checkpoint.exists():
            self._logger.info(f"Downloading LoFTR {self._weights_type} weights to {checkpoint}")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(weights_url, str(checkpoint))


        self._logger.info(f"Initializing LoFTR from {LOFTR_ROOT} onto {self._device}")
        model = LoFTRModel(config=default_cfg)
        ckpt_data = torch.load(str(checkpoint), map_location='cpu')
        model.load_state_dict(ckpt_data['state_dict'])

        self._model = model.to(self._device).eval()
        self._logger.info("LoFTR successfully loaded.")
        return {'model': self._model}
