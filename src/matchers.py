from abc import ABC, abstractmethod
import cv2 as cv


class BaseMatcher(ABC):
    @abstractmethod
    def match(self, img1, img2, kp1,
              kp2, des1, des2, method):
        pass

    @abstractmethod
    def create_matcher(self, method):
        pass

    @abstractmethod
    def knn_match(self, img1, img2, kp1,
                  kp2, des1, des2, method):
        pass


class BFMatcher(BaseMatcher):
    def create_matcher(self, method):
        if method == "orb":
            bf = cv.BFMatcher(cv.NORM_HAMMING)
        else:
            bf = cv.BFMatcher()

        return bf

    def match(self, img1, img2, kp1,
              kp2, des1, des2, method):
        bf = self.create_matcher(method)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        res = cv.drawMatches(img1, kp1, img2, kp2, matches[:20], None,
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return res

    def knn_match(self, img1, img2, kp1,
                  kp2, des1, des2, method):
        bf = self.create_matcher(method)
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        res = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None,
                                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return res


class FLANNMatcher(BaseMatcher):
    def create_matcher(self, method):
        if method == "orb":
            flann_index_lsh = 6
            index_params = dict(algorithm=flann_index_lsh,
                                table_number=6,
                                key_size=12,
                                multi_probe_level=1)
        else:
            flann_index_kdtree = 1
            index_params = dict(algorithm=flann_index_kdtree, trees=5)

        search_params = dict(checks=50)

        flann = cv.FlannBasedMatcher(index_params, search_params)
        return flann

    def match(self, img1, img2, kp1,
              kp2, des1, des2, method):
        flann = self.create_matcher(method)
        matches = flann.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        res = cv.drawMatches(img1, kp1, img2, kp2, matches[:20], None,
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return res

    def knn_match(self, img1, img2, kp1,
                  kp2, des1, des2, method):
        flann = self.create_matcher(method)
        matches = flann.knnMatch(des1, des2, k=2)
        matches_mask = [[0, 0] for i in range(len(matches))]
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matches_mask[i] = [1, 0]

        draw_params = dict(matchesMask=matches_mask,
                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        res = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
        return res


class Matchers:
    _matchers = {
        'bf': BFMatcher,
        'flann': FLANNMatcher
    }

    def select_matcher(self, matcher):
        matcher_class = self._matchers.get(matcher.lower())
        if not matcher_class:
            raise ValueError(f"Matcher '{matcher}' not found ")
        return matcher_class()

    def apply_matcher(self, img1, img2, kp1,
                      kp2, des1, des2, method, matcher, knn):
        matcher_instance = self.select_matcher(matcher)
        if knn:
            result = matcher_instance.knn_match(img1, img2, kp1,
                                                kp2, des1, des2, method)
        else:
            result = matcher_instance.match(img1, img2, kp1,
                                            kp2, des1, des2, method)
        return result
