import pytest
import cv2 as cv
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from logging import Logger

from src.detectors import Detector, SIFTDetector, OpenCVDetector


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


@pytest.fixture
def load_img():
    def _load(name: str):
        path = Path(__file__).parent.parent / "test_data" / name
        img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)

        if img is None:
            pytest.fail(f"Failed to load image. On path: {path.absolute()}")
        return img
    return _load

class TestDetectorRegistry:
    def test_registration_completeness(self):
        expected_algos = {"sift", "orb", "akaze", "fast", "brisk", "kaze"}
        assert expected_algos.issubset(Detector._METHODS.keys())

    def test_internal_classes_not_registered(self):
        assert "opencv" not in Detector._METHODS
        assert "detector" not in Detector._METHODS

    def test_factory_creation_types(self, mock_logger):
        detector = Detector.create("sift", mock_logger)
        assert isinstance(detector, SIFTDetector)
        assert isinstance(detector, OpenCVDetector)


class TestDetection:
    @pytest.mark.parametrize("method_name", Detector._METHODS.keys())
    def test_all_methods_return_valid_tuple(self, method_name, mock_logger, load_img):
        img = load_img("box.png")
        detector = Detector.create(method_name, mock_logger)
        kp = detector.detect(img)

        assert isinstance(kp, tuple)
        if kp:
            assert isinstance(kp[0], cv.KeyPoint)

    def test_reproducibility(self, mock_logger, load_img):
        img = load_img("box.png")
        detector = Detector.create("orb", mock_logger)

        kp1 = detector.detect(img)
        kp2 = detector.detect(img)

        assert len(kp1) == len(kp2)
        assert kp1[0].pt == kp2[0].pt

    def test_orb_nfeatures_parameter(self, mock_logger, load_img):
        img = load_img("box_in_scene.png")
        limit = 30

        detector = Detector.create("orb", mock_logger, nfeatures=limit)
        kp = detector.detect(img)

        assert len(kp) <= limit

    def test_empty_image_logging(self, mock_logger):
        detector = Detector.create("sift", mock_logger)
        black_img = np.zeros((100, 100), dtype=np.uint8)

        detector.detect(black_img)
        mock_logger.warning.assert_called()

    def test_compare_box_and_scene(self, mock_logger, load_img):
        detector = Detector.create("sift", mock_logger)

        kp_box = detector.detect(load_img("box.png"))
        kp_scene = detector.detect(load_img("box_in_scene.png"))
        assert len(kp_scene) > len(kp_box)

    def test_invalid_input_none(self, mock_logger):
        detector = Detector.create("sift", mock_logger)
        kp = detector.detect(None)

        assert kp == ()
        assert mock_logger.error.called