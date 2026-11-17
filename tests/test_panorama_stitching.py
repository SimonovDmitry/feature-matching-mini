import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from stitching.panorama_stitcher import PanoramaStitcher
from stitching.read_write_data import YamlReader


def test_src_sift_stitches_overlapping_images_and_drops_disconnected_frame():
    logger = logging.getLogger(__name__)
    config_path = (Path(__file__).resolve().parents[1] / 'samples' / 'configs'
                   / 'detailed_stitching_src_opencv_config.yaml')
    config = YamlReader(logger, config_path).get_config()[0]
    rng = np.random.default_rng(155)
    texture = rng.integers(0, 255, (160, 360, 3), dtype=np.uint8)
    canvas = cv2.resize(texture, (1080, 480))
    left = canvas[:, :640].copy()
    right = canvas[:, 220:860].copy()
    unrelated = np.full_like(left, 127)
    stitcher = PanoramaStitcher.create(logger, config['stitcher_type'], config)

    baseline = stitcher.stitch([left, right])
    panorama = stitcher.stitch([unrelated, left, right])

    assert baseline.shape[1] > left.shape[1]
    assert panorama.shape == pytest.approx(baseline.shape, abs=4)
    assert stitcher.perf_report['num_images'] == 2

    with pytest.raises(ValueError, match='at least two overlapping images'):
        stitcher.stitch([unrelated, unrelated])
