import torch
import numpy as np
import tensorflow as tf
import sys
from pathlib import Path

OMNIGLUE_ROOT = str(Path(__file__).parent.parent / "omniglue")

if OMNIGLUE_ROOT not in sys.path:
    sys.path.append(OMNIGLUE_ROOT)

from src.matchers import Matcher
from src.converter import Converter
from omniglue.src.omniglue import utils as omniglue_utils


class OmniGlueMatcher(Matcher):
    _shared_matcher = {}

    def __init__(self, logger, matcher_name, descriptor_name, config=None):
        Matcher.__init__(self, logger, matcher_name, descriptor_name)

        if config is None:
            config = {}

        og_export = config.pop('og_export', '/feature-matching-mini/weights/omniglue/og_export')
        if og_export is None:
            raise ValueError("OmniGlueMatcher requires 'og_export' in config")

        self._match_threshold = config.pop('match_threshold', 0.005)
        self._logger = logger

        if config:
            self._logger.warning(f"OmniGlueMatcher: unknown config keys ignored: {list(config.keys())}")


        if og_export not in OmniGlueMatcher._shared_matcher:
            self._logger.info(f"Loading OmniGlue matching network from {og_export}")
            OmniGlueMatcher._shared_matcher[og_export] = tf.saved_model.load(og_export)
        self._matcher = OmniGlueMatcher._shared_matcher[og_export]

    def _init_matcher(self):
        pass

    def _construct_inputs(self, features0, features1):

        kp0 = Converter.to_numpy(features0.get('kp_np', features0.get('keypoints')))
        kp1 = Converter.to_numpy(features1.get('kp_np', features1.get('keypoints')))
        des0 = Converter.to_numpy(features0['descriptors'])
        des1 = Converter.to_numpy(features1['descriptors'])
        scores0 = Converter.to_numpy(features0['scores'])
        scores1 = Converter.to_numpy(features1['scores'])
        des0_dino = features0['des_dino']
        des1_dino = features1['des_dino']

        return {
            'keypoints0': tf.convert_to_tensor(np.expand_dims(kp0, axis=0), dtype=tf.float32),
            'keypoints1': tf.convert_to_tensor(np.expand_dims(kp1, axis=0), dtype=tf.float32),
            'descriptors0': tf.convert_to_tensor(np.expand_dims(des0, axis=0), dtype=tf.float32),
            'descriptors1': tf.convert_to_tensor(np.expand_dims(des1, axis=0), dtype=tf.float32),
            'scores0': tf.convert_to_tensor(np.expand_dims(np.expand_dims(scores0, axis=0), axis=-1), dtype=tf.float32),
            'scores1': tf.convert_to_tensor(np.expand_dims(np.expand_dims(scores1, axis=0), axis=-1), dtype=tf.float32),
            'descriptors0_dino': tf.expand_dims(des0_dino, axis=0),
            'descriptors1_dino': tf.expand_dims(des1_dino, axis=0),
            'width0': tf.convert_to_tensor([features0['width']], dtype=tf.int32),
            'width1': tf.convert_to_tensor([features1['width']], dtype=tf.int32),
            'height0': tf.convert_to_tensor([features0['height']], dtype=tf.int32),
            'height1': tf.convert_to_tensor([features1['height']], dtype=tf.int32),
        }

    def match(self, features0, features1):
        kp0 = features0['kp_np']
        kp1 = features1['kp_np']
        scores0 = np.asarray(features0['scores'])
        scores1 = np.asarray(features1['scores'])

        if kp0.shape[0] == 0 or kp1.shape[0] == 0:
            return {'matches': (), 'scores': ()}

        inputs = self._construct_inputs(features0, features1)
        og_outputs = self._matcher.signatures['serving_default'](**inputs)
        soft_assignment = og_outputs['soft_assignment'][:, :-1, :-1]

        match_matrix = (omniglue_utils.soft_assignment_to_match_matrix(soft_assignment, self._match_threshold)
                        .numpy().squeeze())
        match_indices = np.argwhere(match_matrix)

        keep = []
        for i in range(match_indices.shape[0]):
            m = match_indices[i, :]
            if scores0[m[0]] > 0.0 and scores1[m[1]] > 0.0:
                keep.append(i)

        match_indices = match_indices[keep]
        confidences = []
        for m in match_indices:
            confidences.append(float(soft_assignment[0, m[0], m[1]]))

        return {
            'matches': torch.from_numpy(match_indices).to(torch.long),
            'scores': torch.from_numpy(np.array(confidences)).to(torch.float)
        }
