import sys
from pathlib import Path
import cv2 as cv
import torch
import numpy as np

from src.backend.io_adapter_base import IOAdapter
from src.backend.torch.io_adapter.io_adapter import TorchIOAdapter
from src.backend.torch.utils import image_to_tensor, features_to_cv

D2_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "3rdparty" / "d2net"
if str(D2_ROOT) not in sys.path:
    sys.path.insert(0, str(D2_ROOT))

from lib.utils import preprocess_image


@IOAdapter.register("d2net_torch")
class D2NetTorchIOAdapter(TorchIOAdapter):
    def preprocess(self, inputs):
        img = inputs.pop('image')
        img = image_to_tensor(img)

        if torch.is_tensor(img):
            img_np = img.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
            if img_np.max() <= 1.0:
                img_np = (img_np * 255)
        else:
            img_np = np.array(img)

        img = preprocess_image(img_np, preprocessing='caffe')
        return {'image': img, **inputs}

    def postprocess(self, outputs):
        keypoints = outputs.get('kp')
        descriptors = outputs.get('des')
        scores = outputs.get('sc')

        keypoints_np = keypoints[:, [1, 0]].astype(np.float32)
        if keypoints_np.ndim == 3:
            keypoints_np = keypoints_np.reshape(-1, 2)

        keypoints_np = keypoints_np.astype(np.float32)

        keypoints = cv.KeyPoint_convert(keypoints_np)
        keypoints = np.asarray(keypoints)
        descriptors = descriptors

        result = {'kp': keypoints, 'des': descriptors, 'sc': scores}
        return result
