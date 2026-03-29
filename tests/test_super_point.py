import pytest
import cv2 as cv
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from logging import Logger

from src.super_point import SuperPoint
from src.detectors import Detector
from src.descriptors import Descriptor


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


@pytest.fixture
def load_img():
    def _load(name, color=True):
        path = Path(__file__).parent.parent / "test_data" / name
        mode = cv.IMREAD_COLOR if color else cv.IMREAD_GRAYSCALE
        img = cv.imread(str(path), mode)
        if img is None:
            pytest.skip(f"Test image not found at {path}")
        return img

    return _load


@pytest.fixture
def random_img():
    return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)


@pytest.fixture
def sp_instance(mock_logger):
    return SuperPoint("superpoint", logger=mock_logger)


class TestSuperPointRegistration:
    def test_registered_in_factory(self):
        assert "superpoint" in Detector._METHODS
        assert "superpoint" in Descriptor._METHODS

    def test_factory_creation(self, mock_logger):
        obj = Detector.create("superpoint", mock_logger)
        assert isinstance(obj, SuperPoint)
        assert obj.default_norm == cv.NORM_L2


class TestSuperPointSingleton:
    def test_shared_model_weights(self, mock_logger):

        sp1 = SuperPoint("superpoint", logger=mock_logger)
        sp2 = SuperPoint("superpoint", logger=mock_logger)

        assert sp1._model is sp2._model
        assert sp1._processor is sp2._processor

    def test_instance_parameter_independence(self, mock_logger):

        sp_sensitive = SuperPoint("sp1", mock_logger, threshold=0.001)
        sp_strict = SuperPoint("sp2", mock_logger, threshold=0.1)

        assert sp_sensitive._threshold == 0.001
        assert sp_strict._threshold == 0.1
        assert sp_sensitive._threshold != sp_strict._threshold

    def test_eval_mode_persistence(self, mock_logger):
        sp = SuperPoint("sp", mock_logger)
        assert not sp._model.training

    def test_shared_model_after_deletion(self, mock_logger, random_img):
        sp1 = SuperPoint("sp1", mock_logger)
        sp2 = SuperPoint("sp2", mock_logger)

        del sp1
        kp = sp2.detect(random_img)
        assert isinstance(kp, list)

    def test_device_consistency(self, mock_logger):
        sp1 = SuperPoint("sp1", mock_logger)
        model_device = next(sp1._model.parameters()).device
        assert sp1._device.type == model_device.type


class TestSuperPointInference:
    def test_detect_returns_keypoints(self, sp_instance, load_img):
        img = load_img("box.png")
        kp = sp_instance.detect(img)

        assert isinstance(kp, list)
        if len(kp) > 0:
            assert isinstance(kp[0], cv.KeyPoint)
            h, w = img.shape[:2]
            assert 0 <= kp[0].pt[0] <= w
            assert 0 <= kp[0].pt[1] <= h

    def test_detect_and_compute_consistency(self, sp_instance, load_img):
        img = load_img("box.png")
        kp, des = sp_instance.detectAndCompute(img)

        assert len(kp) == len(des)
        if len(des) > 0:
            assert des.dtype == np.float32
            assert des.shape[1] == 256

    def test_caching_mechanism(self, sp_instance, load_img):
        img = load_img("box.png")
        kp1, des1 = sp_instance.detectAndCompute(img)
        kp2, des2 = sp_instance.detectAndCompute(img)

        assert kp1 is kp2, "Keypoints list was recreated instead of using cache!"
        assert des1 is des2, "Descriptors array was recreated instead of using cache!"

        assert id(kp1) == id(kp2)
        assert id(des1) == id(des2)

    def test_cache_invalidation_on_shape_change(self, sp_instance, load_img):
        img = load_img("box.png")
        kp1, des1 = sp_instance.detectAndCompute(img)

        img_resized = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        kp2, des2 = sp_instance.detectAndCompute(img_resized)

        assert kp1 is not kp2
        assert des1 is not des2

    def test_compute_with_internal_keypoints(self, sp_instance, load_img):
        img = load_img("box.png")

        kp_detected = sp_instance.detect(img)
        kp_computed, des_computed = sp_instance.compute(img, kp_detected)

        assert len(kp_computed) == len(kp_detected)
        assert len(des_computed) == len(kp_detected)

    def test_compute_with_external_keypoints(self, sp_instance, load_img, mock_logger):
        img = load_img("box.png")

        fake_kp = [
            cv.KeyPoint(10, 10, 10),
            cv.KeyPoint(20, 20, 10),
            cv.KeyPoint(30, 30, 10)
        ]
        kp_res, des_res = sp_instance.compute(img, fake_kp)
        mock_logger.warning.assert_called()

        assert len(kp_res) != len(fake_kp)
        assert len(kp_res) == len(des_res)


class TestSuperPointRobustness:
    def test_invalid_input_none(self, sp_instance, mock_logger):
        kp, des = sp_instance._forward(None)
        assert kp == ()
        assert des == ()
        mock_logger.error.assert_called_with("Input image is None. Detection aborted.")

    def test_black_image(self, sp_instance, mock_logger):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        kp = sp_instance.detect(img)

        assert isinstance(kp, list)
        if len(kp) == 0:
            mock_logger.warning.assert_called()

    def test_compute_with_external_kp(self, sp_instance, load_img, mock_logger):
        img = load_img("box.png")
        external_kp = [cv.KeyPoint(x=10, y=10, size=10)]

        kp, des = sp_instance.compute(img, external_kp)
        mock_logger.warning.assert_called()
        assert len(kp) != len(external_kp)

    def test_very_small_image(self, sp_instance):
        tiny_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        try:
            kp = sp_instance.detect(tiny_img)
            assert isinstance(kp, list)
        except Exception as e:
            pytest.fail(f"SuperPoint failed on tiny image: {e}")

    def test_high_noise_image(self, sp_instance):
        noise = np.random.randint(0, 2, (200, 200, 3), dtype=np.uint8) * 255
        kp = sp_instance.detect(noise)
        assert isinstance(kp, list)

    def test_non_contiguous_array(self, sp_instance):
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        sliced_img = img[::2, ::2, :]

        kp = sp_instance.detect(sliced_img)
        assert isinstance(kp, list)
