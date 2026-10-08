import sys
from pathlib import Path
from src.backend.io_adapter_base import IOAdapter
from src.backend.torch.io_adapter.io_adapter import TorchIOAdapter
from src.backend.torch.utils import features_to_tensor, matches_to_cv

LIGHTGLUE_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "3rdparty" / "lightglue"
if str(LIGHTGLUE_ROOT) not in sys.path:
    sys.path.insert(0, str(LIGHTGLUE_ROOT))

from lightglue.utils import rbd


@IOAdapter.register("lightglue_torch")
class LightGlueTorchIOAdapter(TorchIOAdapter):
    def preprocess(self, inputs):
        keypoints0 = inputs['features0'].pop('keypoints')
        descriptors0 = inputs['features0'].pop('descriptors')
        scores0 = inputs['features0'].pop('scores')
        data0 = {'keypoints': keypoints0, 'descriptors': descriptors0, 'scores': scores0}

        keypoints1 = inputs['features1'].pop('keypoints')
        descriptors1 = inputs['features1'].pop('descriptors')
        scores1 = inputs['features1'].pop('scores')
        data1 = {'keypoints': keypoints1, 'descriptors': descriptors1, 'scores': scores1}

        features0 = features_to_tensor(data0)
        features1 = features_to_tensor(data1)

        return {
            "image0": {
                "keypoints": features0.pop('keypoints').unsqueeze(0).to(self._device),
                "descriptors": features0.pop('descriptors').unsqueeze(0).to(self._device),
                "keypoint_scores": features0.pop('scores').unsqueeze(0).to(self._device),
                **inputs['features0']
            },
            "image1": {
                "keypoints": features1.pop('keypoints').unsqueeze(0).to(self._device),
                "descriptors": features1.pop('descriptors').unsqueeze(0).to(self._device),
                "keypoint_scores": features1.pop('scores').unsqueeze(0).to(self._device),
                **inputs['features1']
            }
        }

    def postprocess(self, outputs):
        matches = rbd(outputs)
        matches_cv = matches_to_cv(matches)
        return {'matches': matches_cv}
