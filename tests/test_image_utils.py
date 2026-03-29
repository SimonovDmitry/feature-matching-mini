import pytest
import cv2 as cv
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch
import tempfile
import shutil

from src.image_utils import tensor_to_opencv, read_image, save_image, show_image


@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield Path(dir_path)
    shutil.rmtree(dir_path)


@pytest.fixture
def test_image_path(temp_dir):
    img_path = temp_dir / "test_image.jpg"
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    cv.imwrite(str(img_path), test_img)
    return img_path


@pytest.fixture
def sample_numpy():
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_tensor():
    return torch.rand(3, 100, 100)


@pytest.fixture
def sample_tensor_grayscale():
    return torch.rand(1, 100, 100)


class TestTensorToOpenCV:
    def test_standard_tensor(self, sample_tensor):
        result = tensor_to_opencv(sample_tensor)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_grayscale_tensor(self, sample_tensor_grayscale):
        result = tensor_to_opencv(sample_tensor_grayscale)
        assert result.shape == (100, 100, 1)
        assert result.dtype == np.uint8

    def test_2d_tensor(self):
        tensor = torch.rand(100, 100)
        result = tensor_to_opencv(tensor)
        assert result.shape == (100, 100, 1)

    def test_normalization(self):
        tensor = torch.tensor([0.0, 0.5, 1.0]).reshape(3, 1, 1)
        result = tensor_to_opencv(tensor)
        assert result[0, 0, 0] == 0
        assert result[0, 0, 1] == 127
        assert result[0, 0, 2] == 255


class TestReadImage:
    def test_read_numpy(self, test_image_path):
        img = read_image(test_image_path, input_type='numpy')
        assert isinstance(img, np.ndarray)
        assert img.shape == (100, 100, 3)

    def test_read_tensor(self, test_image_path):
        img = read_image(test_image_path, input_type='tensor')
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 100, 100)
        assert img.max() <= 1.0
        assert img.min() >= 0.0

    def test_read_none_path(self):
        with pytest.raises(ValueError, match="Empty path"):
            read_image(None)

    def test_read_nonexistent_path(self):
        with pytest.raises(ValueError, match="Incorrect path"):
            read_image("/nonexistent.jpg")


class TestSaveImage:
    def test_save_numpy(self, sample_numpy, temp_dir):
        save_path = temp_dir / "output.jpg"
        result = save_image(sample_numpy, save_path, input_type='numpy')

        assert result is True
        assert save_path.exists()

    def test_save_tensor(self, sample_tensor, temp_dir):
        save_path = temp_dir / "output.jpg"
        result = save_image(sample_tensor, save_path, input_type='tensor')

        assert result is True
        assert save_path.exists()

    def test_save_creates_directory(self, sample_numpy, temp_dir):
        save_path = temp_dir / "nested" / "deep" / "output.jpg"
        result = save_image(sample_numpy, save_path, input_type='numpy')

        assert result is True
        assert save_path.exists()

    def test_save_none_image(self):
        with pytest.raises(ValueError, match="Empty image"):
            save_image(None, "output.jpg")

    def test_save_none_path(self, sample_numpy):
        with pytest.raises(ValueError, match="Empty path"):
            save_image(sample_numpy, None)


class TestShowImage:
    @patch('cv2.namedWindow')
    @patch('cv2.resizeWindow')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    @patch('cv2.getWindowImageRect')
    def test_show_numpy(self, mock_get_rect, mock_destroy, mock_wait,
                        mock_imshow, mock_resize, mock_named, sample_numpy):
        mock_get_rect.return_value = (0, 0, 1920, 1080)
        mock_wait.return_value = 27

        show_image(sample_numpy, title="Test", input_type='numpy')

        mock_imshow.assert_called_once()
        mock_wait.assert_called_once_with(0)

    @patch('cv2.namedWindow')
    @patch('cv2.resizeWindow')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    @patch('cv2.getWindowImageRect')
    def test_show_tensor(self, mock_get_rect, mock_destroy, mock_wait,
                         mock_imshow, mock_resize, mock_named, sample_tensor):
        mock_get_rect.return_value = (0, 0, 1920, 1080)
        mock_wait.return_value = 27

        show_image(sample_tensor, title="Test", input_type='tensor')

        mock_imshow.assert_called_once()

    def test_show_none_image(self):
        with pytest.raises(ValueError, match="Empty image"):
            show_image(None)


class TestIntegration:
    def test_read_save_roundtrip(self, test_image_path, temp_dir):
        img = read_image(str(test_image_path), input_type='numpy')

        save_path = temp_dir / "roundtrip.jpg"
        save_image(img, save_path, input_type='numpy')

        img2 = read_image(save_path, input_type='numpy')

        assert img.shape == img2.shape
        np.testing.assert_array_equal(img, img2)

    def test_numpy_tensor_consistency(self, test_image_path):
        img_np = read_image(test_image_path, input_type='numpy')
        img_tensor = read_image(test_image_path, input_type='tensor')

        img_back = tensor_to_opencv(img_tensor)

        assert img_np.shape == img_back.shape
        diff = np.abs(img_np.astype(np.float32) - img_back.astype(np.float32))
        assert diff.mean() < 5.0
