import pytest
import cv2 as cv
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from logging import Logger

from src.matchers import Matcher
from src.feature_matcher import FeatureMatcherCV2


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


class TestFeatureMatcherInit:
    def test_init_params(self, mock_logger):
        matcher = FeatureMatcherCV2(
            logger=mock_logger,
            detector='orb',
            descriptor='orb',
            matcher='bf',
            matcher_mode='knn'
        )
        assert matcher._detector == 'orb'
        assert matcher._descriptor == 'orb'
        assert matcher._matcher == 'bf'
        assert matcher._matcher_mode == 'knn'


class TestFeatureMatcherMatch:
    @pytest.mark.parametrize("det, des, mat", [
        ('sift', 'sift', 'bf'),
        ('orb', 'orb', 'bf'),
        ('sift', 'brisk', 'bf')
    ])
    def test_match_full_pipeline_success(self, det, des, mat, mock_logger, load_img):
        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")

        matcher_cv = FeatureMatcherCV2(
            logger=mock_logger,
            detector=det,
            descriptor=des,
            matcher=mat,
            matcher_mode='simple'
        )

        kp1, kp2, matches = matcher_cv.match(img1, img2)

        assert isinstance(kp1, (list, tuple))
        assert isinstance(kp2, (list, tuple))
        assert isinstance(matches, (list, tuple))
        assert len(matches) > 0

    def test_match_reproducibility(self, mock_logger, load_img):
        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")
        matcher_cv = FeatureMatcherCV2(logger=mock_logger)

        res1 = matcher_cv.match(img1, img2)
        res2 = matcher_cv.match(img1, img2)
        assert len(res1[2]) == len(res2[2])


class TestFeatureMatcherCompatibility:
    valid_combinations = [
        (det, des, mat, mode)
        for det, descriptors in FeatureMatcherCV2._DETECTOR_DESCRIPTOR_COMPATIBILITY.items()
        for des in descriptors
        for mat in Matcher._METHODS.keys()
        for mode in Matcher._MODES.keys()
    ]

    @pytest.mark.parametrize("det, des, mat, mode", valid_combinations)
    def test_known_compatible_pairs(self, det, des, mat, mode, mock_logger, load_img):
        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")

        matcher_cv = FeatureMatcherCV2(
            logger=mock_logger,
            detector=det,
            descriptor=des,
            matcher=mat,
            matcher_mode=mode
        )

        try:
            kp1, kp2, matches = matcher_cv.match(img1, img2)
            assert kp1 is not None, f"Detector {det} failed"
            assert kp2 is not None, f"Detector {det} failed"
            assert isinstance(matches, (list, tuple)), f"Matches error in {mat} ({mode})"

            if mode == 'knn' and len(matches) > 0:
                assert isinstance(matches[0], (list, tuple)), "KNN matches should be list of lists"

        except Exception as e:
            pytest.fail(
                f"FAILED COMBINATION: \n"
                f"Detector: {det} | Descriptor: {des} | Matcher: {mat} | Mode: {mode}\n"
                f"Error: {e}"
            )

    def test_msd_sift_incompatibility(self, mock_logger, load_img):
        img = load_img("box.png")
        matcher_cv = FeatureMatcherCV2(
            logger=mock_logger,
            detector='msd',
            descriptor='sift'
        )
        with pytest.raises(Exception):
            matcher_cv.match(img, img)


class TestFeatureMatcherRobustness:
    def test_match_raises_value_error_on_empty_image(self, mock_logger):
        matcher_cv = FeatureMatcherCV2(logger=mock_logger)
        empty_img = np.zeros((10, 10), dtype=np.uint8)

        with pytest.raises(ValueError, match="Failed to detect key points"):
            matcher_cv.match(empty_img, empty_img)

    def test_match_with_invalid_descriptor_input(self, mock_logger):
        matcher_cv = FeatureMatcherCV2(logger=mock_logger)

        with pytest.raises(Exception):
            matcher_cv.match(None, None)


class TestFeatureMatcherVisualization:
    def test_visualize_matches_shape(self, mock_logger, load_img):
        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")

        matcher_cv = FeatureMatcherCV2(logger=mock_logger, matcher_mode='simple')
        kp1, kp2, matches = matcher_cv.match(img1, img2)
        res_img = matcher_cv.visualize_matches(img1, kp1, img2, kp2, matches)

        assert res_img.shape[0] >= max(img1.shape[0], img2.shape[0])
        assert res_img.shape[1] == img1.shape[1] + img2.shape[1]

    def test_visualize_no_matches(self, mock_logger, load_img):
        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")

        matcher_cv = FeatureMatcherCV2(logger=mock_logger)
        res_img = matcher_cv.visualize_matches(img1, [], img2, [], [])

        assert res_img.shape[1] == img1.shape[1] + img2.shape[1]
        assert mock_logger.warning.called

    def test_visualize_knn_mode(self, mock_logger, load_img):
        img1 = load_img("box.png")
        img2 = load_img("box_in_scene.png")

        matcher_cv = FeatureMatcherCV2(logger=mock_logger, matcher_mode='knn')
        kp1, kp2, matches = matcher_cv.match(img1, img2)

        res_img = matcher_cv.visualize_matches(img1, kp1, img2, kp2, matches)
        assert res_img is not None


class TestFeatureMatcherInvalidInputs:
    def test_raises_value_error_on_invalid_detector(self, mock_logger):
        with pytest.raises(ValueError, match="Detector 'invalid_det' not found"):
            FeatureMatcherCV2(logger=mock_logger, detector='invalid_det').match(
                np.zeros((100, 100), dtype=np.uint8),
                np.zeros((100, 100), dtype=np.uint8)
            )

    def test_raises_value_error_on_invalid_matcher(self, mock_logger):
        matcher_cv = FeatureMatcherCV2(logger=mock_logger, matcher='unknown_matcher')
        img = np.ones((100, 100), dtype=np.uint8) * 50
        with pytest.raises(ValueError):
            matcher_cv.match(img, img)
