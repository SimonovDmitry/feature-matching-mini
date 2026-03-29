import torch

class LightGlueMatcher():
    _SUPPORTED_EXTRACTORS = {
        'superpoint': SuperPoint,
        'disk': DISK,
        'sift': SIFT,
        'aliked': ALIKED
    }

    def __init__(self, logger, extractor_type='superpoint', device='cpu',
                 max_num_keypoints=2048):
        self.logger = logger
        self.device = torch.device(device)
        self.extractor_type = extractor_type
        ExtractorClass = self._SUPPORTED_EXTRACTORS.get(extractor_type.lower())
        if not ExtractorClass:
            raise ValueError(f"Extractor '{extractor_type}' not found ")
        self.extractor = ExtractorClass(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features=extractor_type).eval().to(self.device)

    @staticmethod
    def create(extractor_type='superpoint', logger=None, device='cuda', max_num_keypoints=2048):
        return LightGlueMatcher(logger, extractor_type, device, max_num_keypoints)

    def match(self, image0, image1):
        feats0 = self.extractor.extract(image0.to(self.device))
        feats1 = self.extractor.extract(image1.to(self.device))
        matches01 = matcher({"image0": feats0, "image1": feats1})
        matches01 = rbd(matches01)
        matches = matches01["matches"]
        return matches