import torch

from src.matchers import Matcher
from lightglue import LightGlue
from lightglue.utils import rbd


class LightGlueMatcher(Matcher):
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
        self._extractor_name = descriptor_name.replace('_lightglue', '').lower()
        self._matcher = LightGlue(features=self._extractor_name, **config).eval().to(self._device)

    def _init_matcher(self):
        pass

    def match(self, features0, features1):
        with torch.no_grad():
            input_dict = {"image0": features0, "image1": features1}
            matches01 = self._matcher(input_dict)
            matches01 = rbd(matches01)
            return {'matches': matches01['matches'], 'scores': matches01['scores']}
