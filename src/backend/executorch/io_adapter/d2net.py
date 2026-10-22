import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F

from src.backend.io_adapter_base import IOAdapter
from src.backend.executorch.io_adapter.io_adapter import ExecuTorchIOAdapter
from src.backend.executorch.utils import nms_topk, sample_descriptors

@IOAdapter.register("d2net_executorch")
class D2NetExecuTorchIOAdapter(ExecuTorchIOAdapter):
    def __init__(self, model_name, model_wrapper, logger, config=None):
        super().__init__(model_name, model_wrapper, logger, config)

        self._input_shape = self._config.get("input_shape", (1, 3, 480, 640))
        self._num_keypoints = self._config.get("num_keypoints", 256)
        self._nms_radius = self._config.get("nms_radius", 4)

    def preprocess(self, inputs):
        image = inputs.get("image")

        if not isinstance(image, np.ndarray):
            raise TypeError("Expected image as numpy.ndarray")

        if image.ndim == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if len(self._input_shape) != 4 or self._input_shape[0] != 1:
            raise ValueError("input_shape must be [1, C, H, W]")

        self.orig_h, self.orig_w = image.shape[:2]
        _, channels, input_height, input_width = self._input_shape
        self.input_h = input_height
        self.input_width = input_width

        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        tensor = torch.from_numpy(image).float().div_(255.0)

        if tensor.shape[0] != channels:
            raise ValueError(f"Model expects {channels} channels, got {tensor.shape[0]}")

        tensor = F.interpolate(tensor.unsqueeze(0), size=(input_height, input_width), mode="bilinear",
                               align_corners=False).contiguous()
        return {**inputs, "image": tensor}

    def postprocess(self, outputs):
        dense = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        if not isinstance(dense, torch.Tensor) or dense.ndim != 4:
            raise ValueError("Expected D2-Net dense output with shape [1, C, H, W]")

        score_map = self._d2net_score_map(dense)
        keypoints, scores = nms_topk(score_map, self._num_keypoints, self._nms_radius)

        _, _, feature_height, feature_width = dense.shape
        descriptors = sample_descriptors(dense, keypoints, feature_height, feature_width)

        if hasattr(self, "orig_w") and hasattr(self, "orig_h"):
            keypoints[:, 0] *= self.orig_w / feature_width
            keypoints[:, 1] *= self.orig_h / feature_height

        keypoints_np = keypoints.detach().cpu().numpy()
        if keypoints_np.ndim == 3:
            keypoints_np = keypoints_np.reshape(-1, 2)
        keypoints_np = keypoints_np.astype(np.float32)
        keypoints = cv.KeyPoint_convert(keypoints_np)
        keypoints = np.asarray(keypoints)

        return {
            "keypoints": keypoints,
            "descriptors": descriptors.detach().cpu().numpy(),
            "scores": scores.detach().cpu().numpy(),
        }

    @staticmethod
    def _d2net_score_map(dense):
        exp_dense = torch.exp(dense - dense.amax(dim=(2, 3), keepdim=True))
        local_sum = F.avg_pool2d(exp_dense, kernel_size=3, stride=1, padding=1) * 9

        alpha = exp_dense / (local_sum + 1e-8)
        channel_max = dense.amax(dim=1, keepdim=True)
        beta = dense / (channel_max + 1e-8)

        return (alpha * beta).amax(dim=1)[0]
