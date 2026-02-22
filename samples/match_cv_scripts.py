import argparse
import logging
import os
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "match_cv.log"),
                            mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger("FeatureMatcher")

class FeatureMatcherCV2():
    def __init__(self, method_name = 'sift'):
        self._method_name = method_name
        self._matcher_name = None
        self._extractor = self._create_extractor()

        self._img1 = None
        self._img2 = None
        self._kp1, self._des1 = None, None
        self._kp2, self._des2 = None, None
        self._matches = []
        self._matches_mask = None

    def _create_extractor(self):
        try:
            method = getattr(cv, f"{self._method_name.upper()}_create")
            logger.info(f"Creating {self._method_name} detector")
            return method()
        except AttributeError:
            raise ValueError(f"Method {self._method_name} not found in OpenCV")

    def load_images(self, img1_path, img2_path):
        self._img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
        self._img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)

        if self._img1 is None or self._img2 is None:
            logger.error(f"Failed to load image for match from {img1_path} and {img2_path}")
            return False

        logger.info(f"Successfully loaded images from {img1_path} and {img2_path}")
        return True

    def _extract_features(self):
        self._kp1, self._des1 = self._extractor.detectAndCompute(self._img1, None)
        self._kp2, self._des2 = self._extractor.detectAndCompute(self._img2, None)

        if self._des1 is None or self._des2 is None:
            raise ValueError(f"No descriptors found to match")
        logger.info("Descriptors extracted successfully")

    def _bf_matcher(self):
        bf = cv.BFMatcher(self._extractor.defaultNorm())

        self._matches = bf.knnMatch(self._des1, self._des2, k=2)
        self._matches_mask = [[0, 0] for _ in range(len(self._matches))]
        self._filter_matches()

    def _flann_matcher(self):
        if self._extractor.defaultNorm() == cv.NORM_L2:
            index_params = {"algorithm": 1, "trees": 5}
            d1, d2 = self._des1.astype(np.float32), self._des2.astype(np.float32)
        else:
            index_params = {"algorithm": 6, "table_number": 6,
                            "key_size": 12, "multi_probe_level": 1}
            d1, d2 = self._des1, self._des2

        flann = cv.FlannBasedMatcher(index_params, {"checks": 50})
        self._matches = flann.knnMatch(d1, d2, k=2)
        self._filter_matches()

    def _filter_matches(self, ratio = 0.75):
        self._matches_mask = [[0, 0] for _ in range(len(self._matches))]

        count_matches = 0
        for i, (m, n) in enumerate(self._matches):
            if m.distance < ratio * n.distance:
                self._matches_mask[i] = [1, 0]
                count_matches += 1
        logger.info(f"Found {count_matches} good matches")

    def run_matching(self, matcher_name = 'bf'):
        self._matcher_name = matcher_name.lower()
        self._extract_features()

        method_name = f"_{self._matcher_name}_matcher"
        matcher_func = getattr(self, method_name, None)
        if matcher_func and callable(matcher_func):
            logger.info(f"Starting {self._matcher_name.upper()} matching via dynamic discovery")
            matcher_func()
        else:
            raise ValueError(f"The {matcher_name} matcher is not implemented")

    def visualize_matching(self, save = False):
        if not self._matches or self._matches_mask is None:
            logger.info("No data to display")
            return

        draw_params = {"matchColor": (0, 255, 0), "singlePointColor": (255, 0, 0),
            "matchesMask": self._matches_mask, "flags": cv.DrawMatchesFlags_DEFAULT}
        res_img = cv.drawMatchesKnn(self._img1, self._kp1, self._img2, self._kp2,
                                 self._matches, None, **draw_params)

        if save:
            save_name = f"match_cv_res_{self._method_name}_{self._matcher_name}.jpg"
            save_path = os.path.join(BASE_DIR, save_name)
            cv.imwrite(save_path, res_img)
            logger.info(f"The result of the match is saved in {save_path}")

        plt.title(f"Detector: {self._method_name} Matcher: {self._matcher_name}")
        plt.imshow(res_img)
        plt.axis('off')
        plt.show()
