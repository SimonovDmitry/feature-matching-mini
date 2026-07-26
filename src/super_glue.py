import torch
import torch.nn.functional as functional
import sys
import numpy as np
from pathlib import Path

SUPERGLUE_REPO_PATH = Path(__file__).parent.parent / "superglue_repo"
if str(SUPERGLUE_REPO_PATH) not in sys.path:
    sys.path.append(str(SUPERGLUE_REPO_PATH))

from superglue_repo.models.superglue import SuperGlue  # noqa: E402
from src.matchers import Matcher  # noqa: E402


class SuperGlueMatcher(Matcher):
    def __init__(self, logger, matcher_name, descriptor_name, config=None):
        Matcher.__init__(self, logger, matcher_name, descriptor_name)
        if config is None:
            config = {}

        device = config.pop('device', None)
        if device is None:
            if torch.cuda.is_available():
                self._device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self._device = torch.device('mps')
            else:
                self._device = torch.device('cpu')
        else:
            self._device = torch.device(device)

        self._logger = logger
        sg_config = {
            'weights': config.pop('weights', 'outdoor'),
            'sinkhorn_iterations': config.pop('sinkhorn_iterations', 20),
            'match_threshold': config.pop('threshold', 0.005),
        }

        self._logger.info(f"Loading SuperGlue ({sg_config.get('weights')}) onto {self._device}")
        self._matcher = SuperGlue(sg_config).to(self._device).eval()

    def _init_matcher(self):
        pass

    def prep(self, feat):
        kps = feat['keypoints']
        des = feat['descriptors']
        scores = feat['scores']

        if not torch.is_tensor(kps):
            kps = torch.from_numpy(kps).float()
        if not torch.is_tensor(des):
            des = torch.from_numpy(des).float()
        if not torch.is_tensor(scores):
            scores = torch.from_numpy(scores).float()

        des = functional.normalize(des, p=2, dim=1)

        data = {
            'keypoints': kps.unsqueeze(0).to(self._device),
            'descriptors': des.T.unsqueeze(0).to(self._device),
            'scores': scores.unsqueeze(0).to(self._device),
        }
        data['image'] = torch.empty(1, 1, feat['height'], feat['width']).to(self._device)
        return data

    def match(self, features0, features1):
        if len(features0['keypoints']) == 0 or len(features1['keypoints']) == 0:
            return {'matches': (), 'scores': ()}

        data0 = self.prep(features0)
        data1 = self.prep(features1)

        input_dict = {
            'keypoints0': data0['keypoints'],
            'keypoints1': data1['keypoints'],
            'descriptors0': data0['descriptors'],
            'descriptors1': data1['descriptors'],
            'scores0': data0['scores'],
            'scores1': data1['scores'],
            'image0': data0['image'],
            'image1': data1['image'],
        }

        with torch.no_grad():
            pred = self._matcher(input_dict)

        matches0 = pred['matches0'][0].cpu().numpy()
        confidences = pred['matching_scores0'][0].cpu().numpy()

        valid = (matches0 > -1) & (matches0 < len(features1['keypoints']))
        idx0 = np.where(valid)[0]
        idx1 = matches0[valid]

        match_indices = np.stack([idx0, idx1], axis=1)
        res_scores = confidences[valid]

        self._logger.info(f"SuperGlue: {len(match_indices)} matches found.")

        return {
            'matches': torch.from_numpy(match_indices).long(),
            'scores': torch.from_numpy(res_scores).float()
        }
