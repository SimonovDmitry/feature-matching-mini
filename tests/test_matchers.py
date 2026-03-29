import pytest
import cv2 as cv
import numpy as np
from unittest.mock import MagicMock
from logging import Logger

from src.descriptors import Descriptor
from src.matchers import Matcher, BFMatcher, FLANNMatcher


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


@pytest.fixture
def mock_descriptor():
    descriptor = MagicMock(spec=Descriptor)
    descriptor.default_norm = cv.NORM_L2
    return descriptor


@pytest.fixture
def mock_descriptor_hamming():
    descriptor = MagicMock(spec=Descriptor)
    descriptor.default_norm = cv.NORM_HAMMING
    return descriptor


@pytest.fixture
def test_descriptors():
    np.random.seed(42)
    des1 = np.random.rand(50, 128).astype(np.float32)
    des2 = np.random.rand(60, 128).astype(np.float32)
    return des1, des2


@pytest.fixture
def test_descriptors_binary():
    np.random.seed(42)
    des1 = np.random.randint(0, 255, (50, 32), dtype=np.uint8)
    des2 = np.random.randint(0, 255, (60, 32), dtype=np.uint8)
    return des1, des2


class TestMatcherRegistry:
    def test_registration_completeness(self):
        expected_matchers = {"bf", "flann"}
        assert expected_matchers.issubset(Matcher._METHODS.keys())

    def test_internal_classes_not_registered(self):
        assert "matcher" not in Matcher._METHODS

    def test_factory_creation_types_bf(self, mock_logger, mock_descriptor):
        matcher = Matcher.create("bf", mock_logger, mock_descriptor, mode="simple")
        assert isinstance(matcher, BFMatcher)
        assert isinstance(matcher, Matcher)

    def test_factory_creation_types_flann(self, mock_logger, mock_descriptor):
        matcher = Matcher.create("flann", mock_logger, mock_descriptor, mode="simple")
        assert isinstance(matcher, FLANNMatcher)
        assert isinstance(matcher, Matcher)


class TestMatcherModes:
    def test_simple_mode_returns_list_of_dmatches(self, mock_logger, mock_descriptor, test_descriptors):
        matcher = BFMatcher(mock_logger, mock_descriptor, mode="simple")
        des1, des2 = test_descriptors
        matches = matcher.match(des1, des2)

        assert isinstance(matches, list)
        if matches:
            assert isinstance(matches[0], cv.DMatch)

    def test_knn_mode_returns_list_of_lists(self, mock_logger, mock_descriptor, test_descriptors):
        matcher = BFMatcher(mock_logger, mock_descriptor, mode="knn")
        des1, des2 = test_descriptors
        matches = matcher.match(des1, des2, k=2)

        assert isinstance(matches, list)
        if matches:
            assert isinstance(matches[0], list)
            assert isinstance(matches[0][0], cv.DMatch)

    def test_knn_returns_k_matches_per_query(self, mock_logger, mock_descriptor, test_descriptors):
        matcher = BFMatcher(mock_logger, mock_descriptor, mode="knn")
        des1, des2 = test_descriptors
        k_count = 3
        matches = matcher.match(des1, des2, k=k_count)

        if matches:
            assert len(matches[0]) == k_count


class TestBFMatcher:
    def test_bf_initialization_with_norm(self, mock_logger, mock_descriptor):
        matcher = BFMatcher(mock_logger, mock_descriptor, mode="simple")
        bf_matcher = matcher._init_matcher()
        assert isinstance(bf_matcher, cv.BFMatcher)

    def test_bf_simple_match_returns_valid_result(self, mock_logger, mock_descriptor, test_descriptors):
        matcher = BFMatcher(mock_logger, mock_descriptor, mode="simple")
        des1, des2 = test_descriptors
        matches = matcher._simple_match(des1, des2)

        assert isinstance(matches, list)
        if matches:
            assert len(matches) == len(des1)

    def test_bf_knn_match_returns_valid_result(self, mock_logger, mock_descriptor, test_descriptors):
        matcher = BFMatcher(mock_logger, mock_descriptor, mode="knn")
        des1, des2 = test_descriptors
        matches = matcher._knn_match(des1, des2, k=2)

        assert isinstance(matches, list)
        if matches:
            assert len(matches) == len(des1)

    def test_bf_reproducibility(self, mock_logger, mock_descriptor, test_descriptors):
        matcher = BFMatcher(mock_logger, mock_descriptor, mode="knn")
        des1, des2 = test_descriptors

        matches1 = matcher.match(des1, des2, k=2)
        matches2 = matcher.match(des1, des2, k=2)

        assert len(matches1) == len(matches2)
        if matches1 and matches2:
            assert matches1[0][0].queryIdx == matches2[0][0].queryIdx
            assert matches1[0][0].trainIdx == matches2[0][0].trainIdx


class TestFLANNMatcher:
    def test_flann_initialization_l2_norm(self, mock_logger, mock_descriptor):
        matcher = FLANNMatcher(mock_logger, mock_descriptor, mode="simple")
        assert matcher.index_params['algorithm'] == 1
        assert 'trees' in matcher.index_params
        assert matcher.index_params['trees'] == 5
        assert 'checks' in matcher.search_params
        assert matcher.search_params['checks'] == 50

    def test_flann_initialization_hamming_norm(self, mock_logger, mock_descriptor_hamming):
        matcher = FLANNMatcher(mock_logger, mock_descriptor_hamming, mode="simple")
        assert matcher.index_params['algorithm'] == 6
        assert 'table_number' in matcher.index_params
        assert 'key_size' in matcher.index_params
        assert matcher.index_params['table_number'] == 6
        assert matcher.index_params['key_size'] == 12
        assert 'multi_probe_level' in matcher.index_params
        assert matcher.index_params['multi_probe_level'] == 1
        assert 'checks' in matcher.search_params
        assert matcher.search_params['checks'] == 50

    def test_flann_initialization_with_custom_params(self, mock_logger, mock_descriptor):
        custom_index = {'algorithm': 1, 'trees': 10}
        custom_search = {'checks': 100}
        matcher = FLANNMatcher(
            mock_logger, mock_descriptor, mode="simple",
            index_params=custom_index, search_params=custom_search)
        assert matcher.index_params['trees'] == 10
        assert matcher.search_params['checks'] == 100

    def test_flann_simple_match_returns_valid_result(self, mock_logger, mock_descriptor, test_descriptors):
        matcher = FLANNMatcher(mock_logger, mock_descriptor, mode="simple")
        des1, des2 = test_descriptors
        matches = matcher._simple_match(des1, des2)

        assert isinstance(matches, list)
        if matches:
            assert isinstance(matches[0], cv.DMatch)

    def test_flann_knn_match_returns_valid_result(self, mock_logger, mock_descriptor, test_descriptors):
        matcher = FLANNMatcher(mock_logger, mock_descriptor, mode="knn")
        des1, des2 = test_descriptors
        matches = matcher._knn_match(des1, des2, k=2)

        assert isinstance(matches, list)
        if matches:
            assert isinstance(matches[0], list)
            assert isinstance(matches[0][0], cv.DMatch)

    def test_flann_with_binary_descriptors(self, mock_logger, mock_descriptor_hamming, test_descriptors_binary):

        matcher = FLANNMatcher(mock_logger, mock_descriptor_hamming, mode="knn")
        des1, des2 = test_descriptors_binary
        matches = matcher.match(des1, des2, k=2)

        assert isinstance(matches, list)
        assert matches is not None


class TestMatcherEdgeCases:
    def test_match_with_empty_descriptors(self, mock_logger, mock_descriptor):
        matcher = BFMatcher(mock_logger, mock_descriptor, mode="simple")
        des1 = np.array([]).reshape(0, 128).astype(np.float32)
        des2 = np.random.rand(10, 128).astype(np.float32)

        matches = matcher.match(des1, des2)
        assert isinstance(matches, list)
        assert len(matches) == 0

    def test_match_with_mismatched_dimensions(self, mock_logger, mock_descriptor):
        matcher = BFMatcher(mock_logger, mock_descriptor, mode="simple")
        des1 = np.random.rand(10, 128).astype(np.float32)
        des2 = np.random.rand(10, 64).astype(np.float32)

        with pytest.raises(Exception):
            matcher.match(des1, des2)
