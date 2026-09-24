from pathlib import Path
import numpy as np
from transformers import AutoImageProcessor, SuperPointForKeypointDetection

from src.dnn_extractors import DNNFeatureExtractors
from src.utils_image import to_numpy_bgr


class SuperPoint(DNNFeatureExtractors):
    _image_processor = None

    def __init__(self, extractor_name, logger, config=None):
        if config is None:
            config = {}

        DNNFeatureExtractors.__init__(self, extractor_name, logger, config)

        checkpoint = config.pop('checkpoint', 'weights/superpoint')
        local_files_only = config.pop('local_files_only', True)
        backend = config.pop('backend', 'torch')
        remote_repo = 'magic-leap-community/superpoint'

        if SuperPoint._model is None:
            local_path = Path(checkpoint)
            if not local_path.exists() or not any(local_path.iterdir()):
                self._logger.warning(f"Local checkpoint {checkpoint} not found or empty.")
                self._logger.info(f"Switching to remote repository: {remote_repo}")
                checkpoint = remote_repo
                local_files_only = False

            try:
                self._logger.info(f'Loading SuperPoint from {checkpoint} (local={local_files_only})')
                SuperPoint._image_processor = AutoImageProcessor.from_pretrained(checkpoint,
                                                                                 local_files_only=local_files_only)
                SuperPoint._model = SuperPointForKeypointDetection.from_pretrained(checkpoint,
                                                                                   local_files_only=local_files_only)
            except Exception as exc:
                self._logger.error(f'Failed to load from {checkpoint}: {exc}')

        self._processor = SuperPoint._image_processor
        self._model = SuperPoint._model

        from src.inference_api import InferenceAPI
        from src.inference_torch import TorchInferenceAPI
        self._inference = InferenceAPI.create(backend, logger, extractor_name, self._model, config=config)

    def _preprocess(self, img):
        if not hasattr(img, 'shape'):
            raise TypeError('SuperPoint expects an image with a shape')

        try:
            from src import utils_torch
            input_type = 'tensor' if utils_torch.is_tensor(img) else 'numpy'
        except ImportError:
            input_type = 'numpy'

        img = to_numpy_bgr(img, input_type=input_type)
        height, width = img.shape[:2]

        inputs = self._processor(img, return_tensors='np')
        return dict(inputs), height, width

    def _forward(self, img):
        if img is None:
            self._logger.error("Input image is None. Detection aborted.")
            return {'keypoints': (), 'descriptors': ()}

        self._logger.info(f"Running inference with {self._detector_name}")
        inputs, height, width = self._preprocess(img)

        try:
            outputs = self._inference.run(inputs=inputs)

            processed = self._processor.post_process_keypoint_detection(outputs, [[height, width]])[0]
            raw_kp = processed['keypoints']
            raw_scores = processed['scores']
            raw_des = processed['descriptors']

            mask = raw_scores > self._threshold
            kp = raw_kp[mask]
            des = raw_des[mask]
            scores = raw_scores[mask]

            if self._nfeatures is not None and len(kp) > self._nfeatures:
                indices = np.argpartition(scores, -self._nfeatures)[-self._nfeatures:]
                indices = indices[np.argsort(scores[indices])[::-1]]

                scores = scores[indices]
                kp = kp[indices]
                des = des[indices]

            extracted = {
                'keypoints': kp,
                'descriptors': des,
                'scores': scores,
                'width': width,
                'height': height
            }
            SuperPoint._extracted_data = extracted

            kp = extracted['keypoints']
            des = extracted['descriptors']

            if len(kp) > 0:
                self._logger.info(f'{self._detector_name} found {len(kp)} points')
            else:
                self._logger.warning(f'{self._detector_name} found 0 points')

            if des is not None:
                self._logger.info(f'{self._descriptor_name} computed {len(des)} descriptors')
            else:
                self._logger.warning(f"{self._descriptor_name} computed 0 descriptors")

            return extracted

        except Exception as e:
            self._logger.warning(f"{self._detector_name} inference failed (likely 0 points): {e}")
            return {'keypoints': (), 'descriptors': (), 'width': width, 'height': height}
