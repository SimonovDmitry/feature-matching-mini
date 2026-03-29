import pytest
import cv2 as cv
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from logging import Logger

from src.detectors import Detector
from src.descriptors import Descriptor, SIFTDescriptor, OpenCVDescriptor


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


@pytest.fixture
def load_img():
    def _load(name):
        path = Path(__file__).parent.parent / "test_data" / name
        img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)

        if img is None:
            pytest.fail(f"Failed to load image. On path: {path.absolute()}")
        return img
    return _load


@pytest.fixture
def get_kp(load_img, mock_logger):
    def _get(img_name, method_name="sift"):
        img = load_img(img_name)
        detector = Detector.create(method_name, mock_logger)
        kp = detector.detect(img)
        return kp

    return _get


class TestDescriptorRegistry:
    def test_registration_completeness(self):
        expected = {"sift", "orb", "akaze", "brisk", "kaze"}
        assert expected.issubset(Descriptor._METHODS.keys())

    def test_internal_classes_not_registered(self):
        assert "opencv" not in Descriptor._METHODS
        assert "descriptor" not in Descriptor._METHODS

    def test_factory_creation_types(self, mock_logger):
        obj = Descriptor.create("sift", mock_logger)
        assert isinstance(obj, SIFTDescriptor)
        assert isinstance(obj, OpenCVDescriptor)


class TestDescriptorFactory:
    def test_factory_creation_types(self, mock_logger):
        obj = Descriptor.create("sift", mock_logger)
        assert isinstance(obj, SIFTDescriptor)
        assert isinstance(obj, OpenCVDescriptor)

    def test_create_unknown_descriptor_raises_error(self, mock_logger):
        with pytest.raises(ValueError, match="Descriptor 'unknown' not found"):
            Descriptor.create("unknown", mock_logger)


class TestDescriptorCompute:
    @pytest.mark.parametrize("method_name", Descriptor._METHODS.keys())
    def test_all_methods_compute_descriptors(self, method_name, mock_logger, load_img, get_kp):
        img = load_img("box.png")
        descriptor = Descriptor.create(method_name, mock_logger)

        test_kp = get_kp("box.png", method_name)[:20]
        kp_out, des = descriptor.compute(img, test_kp)

        assert isinstance(kp_out, (tuple, list))
        if des is not None:
            assert isinstance(des, np.ndarray)
            assert len(des) == len(kp_out)

        assert mock_logger.info.called

    def test_sift_descriptor_size(self, mock_logger, load_img, get_kp):
        img = load_img("box.png")
        descriptor = Descriptor.create("sift", mock_logger)
        _, des = descriptor.compute(img, get_kp("box.png", "sift")[:5])

        assert des is not None
        assert des.shape[1] == 128

    def test_orb_descriptor_size(self, mock_logger, load_img, get_kp):
        img = load_img("box.png")
        descriptor = Descriptor.create("orb", mock_logger)
        _, des = descriptor.compute(img, get_kp("box.png", "orb")[:5])

        assert des is not None
        assert des.shape[1] == 32

    def test_invalid_input_none_image(self, mock_logger, get_kp):
        descriptor = Descriptor.create("sift", mock_logger)
        result = descriptor.compute(None, get_kp("box.png", "sift")[:5])

        assert result == ()
        assert mock_logger.error.called

    def test_empty_keypoints_warning(self, mock_logger, load_img):
        img = load_img("box.png")
        descriptor = Descriptor.create("orb", mock_logger)
        _, des = descriptor.compute(img, ())

        assert des is None or len(des) == 0
        assert mock_logger.warning.called

    def test_reproducibility(self, mock_logger, load_img, get_kp):
        img = load_img("box.png")
        descriptor = Descriptor.create("sift", mock_logger)
        test_kp = get_kp("box.png", "sift")[:10]

        _, des1 = descriptor.compute(img, test_kp)
        _, des2 = descriptor.compute(img, test_kp)
        assert np.array_equal(des1, des2)

    def test_compare_descriptors_on_different_images(self, mock_logger, load_img, get_kp):
        descriptor = Descriptor.create("sift", mock_logger)

        img_box = load_img("box.png")
        kp_box = get_kp("box.png")
        _, des_box = descriptor.compute(img_box, kp_box)

        img_scene = load_img("box_in_scene.png")
        kp_scene = get_kp("box_in_scene.png")
        _, des_scene = descriptor.compute(img_scene, kp_scene)

        assert des_box is not None
        assert des_scene is not None
        assert len(des_scene) > len(des_box)

    def test_descriptor_consistency(self, mock_logger, load_img, get_kp):
        img = load_img("box.png")
        kp = get_kp("box.png", "orb")[:10]

        descriptor = Descriptor.create("orb", mock_logger)
        _, des1 = descriptor.compute(img, kp)
        _, des2 = descriptor.compute(img, kp)
        assert np.array_equal(des1, des2)

    def test_invalid_input_none(self, mock_logger, get_kp):
        descriptor = Descriptor.create("sift", mock_logger)
        kp = get_kp("box.png", "sift")

        result = descriptor.compute(None, kp)
        assert result == ()
        mock_logger.error.assert_called()


class TestDescriptorRobustness:
    def test_invalid_input_none(self, mock_logger, get_kp):
        descriptor = Descriptor.create("sift", mock_logger)
        result = descriptor.compute(None, get_kp("box.png", "sift")[:5])
        assert result == ()
        assert mock_logger.error.called

    def test_keypoints_outside_bounds(self, mock_logger, load_img):
        img = load_img("box.png")
        descriptor = Descriptor.create("sift", mock_logger)
        bad_kp = [cv.KeyPoint(x=10000, y=10000, size=10)]
        kp_out, des = descriptor.compute(img, bad_kp)
        if des is not None and len(des) > 0:
            assert np.all(des == 0) or len(kp_out) == 0
        else:
            assert len(kp_out) == 0 or des is None

    def test_empty_keypoints_warning(self, mock_logger, load_img):
        img = load_img("box.png")
        descriptor = Descriptor.create("orb", mock_logger)
        _, des = descriptor.compute(img, ())
        assert mock_logger.warning.called


class TestDescriptorInvariance:
    def test_brightness_invariance(self, mock_logger, load_img, get_kp):
        img = load_img("box.png")
        bright_img = cv.convertScaleAbs(img, alpha=1.2, beta=30)
        descriptor = Descriptor.create("sift", mock_logger)
        kp = get_kp("box.png", "sift")[:10]
        _, des1 = descriptor.compute(img, kp)
        _, des2 = descriptor.compute(bright_img, kp)
        cos_sim = np.dot(des1[0], des2[0]) / (np.linalg.norm(des1[0]) * np.linalg.norm(des2[0]))
        assert cos_sim > 0.95

    def test_flip_consistency(self, mock_logger, load_img):
        img = load_img("box.png")
        flipped_img = cv.flip(img, 1)
        descriptor = Descriptor.create("orb", mock_logger)
        kp = [cv.KeyPoint(100, 100, 10)]
        flipped_kp = [cv.KeyPoint(img.shape[1] - 100, 100, 10)]
        _, des1 = descriptor.compute(img, kp)
        _, des2 = descriptor.compute(flipped_img, flipped_kp)
        assert des1.shape == des2.shape
