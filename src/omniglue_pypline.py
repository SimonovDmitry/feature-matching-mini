import cv2 as cv
import tensorflow as tf
import torch
import sys
from pathlib import Path

OMNIGLUE_ROOT = str(Path(__file__).parent.parent / "omniglue")

if OMNIGLUE_ROOT not in sys.path:
    sys.path.append(OMNIGLUE_ROOT)

from src.detectors import Detector
from src.descriptors import Descriptor
from src.converter import ImageConverter

from omniglue.src.omniglue import superpoint_extract
from omniglue.src.omniglue import dino_extract


class OmniGlueFeatureExtractor(Detector, Descriptor, register=False):
    _DINO_FEATURE_DIM = 768

    _shared_sp = {}
    _shared_dino = {}
    _is_extracted = False
    _extracted_data = {}

    def __init__(self, extractor_name, logger, config=None):
        Detector.__init__(self, logger, extractor_name)
        Descriptor.__init__(self, logger, extractor_name)

        if config is None:
            config = {}

        sp_export = config.pop('sp_export', '/feature-matching-mini/weights/omniglue/sp_v6')
        dino_export = config.pop('dino_export', '/feature-matching-mini/weights/omniglue/dinov2_vitb14_pretrain.pth')
        if sp_export is None or dino_export is None:
            raise ValueError("OmniGlueExtractor requires both 'sp_export' and 'dino_export' paths in config")

        self._num_features = config.pop('num_features', 1024)
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

        if config:
            self._logger.warning(f"OmniGlueExtractor: unknown config keys ignored: {list(config.keys())}")


        if sp_export not in OmniGlueFeatureExtractor._shared_sp:
            self._logger.info(f"Loading SuperPoint weights from {sp_export}")
            OmniGlueFeatureExtractor._shared_sp[sp_export] = superpoint_extract.SuperPointExtract(sp_export)
        self._sp_extract = OmniGlueFeatureExtractor._shared_sp[sp_export]


        dino_key = (dino_export, self._device.type)
        if dino_key not in OmniGlueFeatureExtractor._shared_dino:
            self._logger.info(f"Loading DINOv2 weights from {dino_export} onto {self._device}")
            dino_model = dino_extract.DINOExtract(dino_export, feature_layer=1)
            dino_model.device = self._device
            dino_model.model = dino_model.model.to(self._device)
            OmniGlueFeatureExtractor._shared_dino[dino_key] = dino_model
        self._dino_extract = OmniGlueFeatureExtractor._shared_dino[dino_key]

    @property
    def default_norm(self):
        return cv.NORM_L2

    def _forward(self, img):
        if img is None:
            self._logger.error("Input image is None. Detection aborted")
            return {'keypoints': (), 'descriptors': (), 'scores': ()}

        img_conv = ImageConverter()

        from_fmt = 'tensor' if torch.is_tensor(img) else 'opencv'
        img_bgr = img_conv.convert(img, from_format=from_fmt, to_format='opencv')
        img_np = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

        height, width = img_np.shape[:2]
        self._logger.info(f"Running inference with {self._detector_name}. Image: {width}x{height}")

        kp, des_sp, scores = self._sp_extract(img_np, num_features=self._num_features)

        if kp.shape[0] == 0:
            self._logger.warning(f"{self._detector_name} found 0 points")
            return {'keypoints': (), 'descriptors': (), 'scores': ()}

        dino_features = self._dino_extract(img_np)
        des_dino = dino_extract.get_dino_descriptors(
            dino_features,
            tf.convert_to_tensor(kp, dtype=tf.float32),
            tf.convert_to_tensor(height, dtype=tf.int32),
            tf.convert_to_tensor(width, dtype=tf.int32),
            OmniGlueFeatureExtractor._DINO_FEATURE_DIM,
        )

        OmniGlueFeatureExtractor._extracted_data = {
            'keypoints': torch.from_numpy(kp).float(),
            'descriptors': torch.from_numpy(des_sp).float(),
            'scores': torch.from_numpy(scores).float(),
            'kp_np': kp,
            'des_dino': des_dino,
            'width': width,
            'height': height,
        }

        self._logger.info(f"{self._detector_name} successfully extracted {kp.shape[0]} points")
        return OmniGlueFeatureExtractor._extracted_data

    def detect(self, img):
        OmniGlueFeatureExtractor._is_extracted = True
        return self._forward(img)

    def compute(self, img, features=None):
        if OmniGlueFeatureExtractor._is_extracted:
            OmniGlueFeatureExtractor._is_extracted = False
            return OmniGlueFeatureExtractor._extracted_data
        else:
            return self._forward(img)

    def detectAndCompute(self, img):
        return self._forward(img)

Detector._METHODS.update({'superpoint_omniglue': OmniGlueFeatureExtractor})
Descriptor._METHODS.update({'superpoint_omniglue': OmniGlueFeatureExtractor})
