import cv2 as cv
import numpy as np
from abc import ABC, abstractmethod


class HPatchesTask(ABC):
    _TASKS = {}
    _img_indices = [2, 3, 4, 5, 6]

    def __init__(self, logger):
        self._logger = logger

    def __init_subclass__(cls, register=True, **kwargs):
        super().__init_subclass__(**kwargs)

        if register:
            key = cls.__name__.replace("Task", "").lower()
            if key:
                HPatchesTask._TASKS[key] = cls

    @classmethod
    def create(cls, task_name, logger, config=None):
        if config is None:
            config = {}

        task_class = cls._TASKS.get(task_name.lower())
        if not task_class:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(cls._TASKS.keys())}")

        return task_class(logger, config)

    @abstractmethod
    def eval_task(self, descriptors, split):
        pass

    @classmethod
    def _tpfp(cls, scores, labels, numpos=None):
        p = int(np.sum(labels))
        n = len(labels) - p

        if numpos is not None:
            assert (numpos >= p), 'numpos smaller that number of positives in labels'
            extra_pos = numpos - p
            p = numpos

            scores = np.hstack((scores, np.repeat(-np.inf, extra_pos)))
            labels = np.hstack((labels, np.repeat(1, extra_pos)))

        perm = np.argsort(-scores, kind='mergesort', axis=0)
        scores = scores[perm]
        stop = np.max(np.where(scores > -np.inf))
        perm = perm[0:stop + 1]
        labels = labels[perm]

        tp = np.hstack((0, np.cumsum(labels == 1)))
        fp = np.hstack((0, np.cumsum(labels == 0)))
        return tp, fp, p, n, perm

    @classmethod
    def _pr(cls, scores, labels, numpos=None):
        [tp, fp, p, n, perm] = cls._tpfp(scores, labels, numpos)

        small = 1e-10
        recall = tp / float(np.maximum(p, small))
        precision = np.maximum(tp, small) / np.maximum(tp + fp, small)
        return precision, recall, np.trapezoid(precision, recall)


class BaseMatchingTask(HPatchesTask, register=False):
    def __init__(self, logger, config):
        super().__init__(logger)
        self._eval_threshold = config.get('eval_threshold', 5.0)

    def _compute_numpos(self, data):
        kp_ref = data['kp_ref']
        kp_tgt = data['kp_tgt']
        H = data['H']
        tgt_shape = data['tgt_shape']

        if len(kp_ref) == 0 or len(kp_tgt) == 0:
            return 0

        pts_ref = np.array([kp.pt for kp in kp_ref], dtype=np.float32).reshape(-1, 1, 2)
        pts_tgt_gt = cv.perspectiveTransform(pts_ref, H).reshape(-1, 2)

        if tgt_shape is not None:
            h, w = tgt_shape[:2]
            valid_mask = ((pts_tgt_gt[:, 0] >= 0) & (pts_tgt_gt[:, 0] < w)
                          & (pts_tgt_gt[:, 1] >= 0) & (pts_tgt_gt[:, 1] < h))
            pts_tgt_gt = pts_tgt_gt[valid_mask]

        if len(pts_tgt_gt) == 0:
            return 0

        pts_tgt = np.array([kp.pt for kp in kp_tgt], dtype=np.float32)

        num_gt = 0
        for pt_proj in pts_tgt_gt:
            pixel_dists = np.linalg.norm(pts_tgt - pt_proj, axis=1)
            if len(pixel_dists) > 0 and np.min(pixel_dists) <= self._eval_threshold:
                num_gt += 1

        return num_gt

    def _get_match_results(self, data):
        kp_ref = data['kp_ref']
        kp_tgt = data['kp_tgt']
        matches = data['matches']
        H = data['H']

        if not matches:
            return None

        pts_ref = np.array([kp_ref[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
        pts_tgt_pred = np.array([kp_tgt[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
        pts_tgt_gt = cv.perspectiveTransform(pts_ref, H)

        pixel_dists = np.linalg.norm(pts_tgt_pred - pts_tgt_gt, axis=2).flatten()
        labels = (pixel_dists <= self._eval_threshold)
        descriptor_dists = np.array([m.distance for m in matches], dtype=np.float32)

        if descriptor_dists.size > 0:
            max_dist = descriptor_dists.max()
            scores = 1.0 - (descriptor_dists / max_dist)
        else:
            scores = np.array([], dtype=np.float32)

        return {
            'labels': labels,
            'distances': pixel_dists,
            'num_kp_ref': len(kp_ref),
            'num_kp_tgt': len(kp_tgt),
            'num_matches': len(matches),
            'scores': scores
        }


class MatchingAPTask(BaseMatchingTask):
    def eval_task(self, matching_data, split):
        self._logger.info(f'Evaluating Feature Matching (mAP) @ {self._eval_threshold}px')
        results = {seq: {} for seq in split}

        for seq in split:
            if seq not in matching_data:
                continue

            for i in self._img_indices:
                data = matching_data[seq].get(i)
                res = self._get_match_results(data) if data else None

                if res is not None:
                    _, _, ap = self._pr(res['scores'], res['labels'], numpos=self._compute_numpos(data))
                    results[seq][i] = {'ap': ap}
        return results

    def report_metrics(self, results, task_name="Feature Matching (mAP)"):
        all_ap_values = [
            img_data['ap']
            for scene_data in results.values()
            for img_data in scene_data.values()
            if 'ap' in img_data
        ]

        if not all_ap_values:
            self._logger.warning("No AP results found to report.")
            return

        mean_total_ap = np.mean(all_ap_values)
        self._logger.info(f"--- {task_name.upper()} ---")
        self._logger.info(f"Mean Total AP: {mean_total_ap:.4f}")


class MatchingScoreTask(BaseMatchingTask):
    def eval_task(self, matching_data, split):
        self._logger.info(f'Evaluating Matching Score & Precision @ {self._eval_threshold}px')
        results = {seq: {} for seq in split}

        for seq in split:
            if seq not in matching_data:
                continue

            for i in self._img_indices:
                data = matching_data[seq].get(i)
                res = self._get_match_results(data) if data else None

                if res is not None:
                    num_inliers = np.sum(res['labels'])
                    results[seq][i] = {
                        'ms': num_inliers / min(res['num_kp_ref'], res['num_kp_tgt']),
                        'prec': num_inliers / res['num_matches'] if res['num_matches'] > 0 else 0
                    }
        return results

    def report_metrics(self, results, task_name="Matching Score & Precision"):
        all_ms_values = [
            img_data['ms']
            for scene_data in results.values()
            for img_data in scene_data.values()
            if 'ms' in img_data
        ]

        if not all_ms_values:
            self._logger.warning("No Matching Score results found to report.")
            return

        all_prec_values = [
            img_data['prec']
            for scene_data in results.values()
            for img_data in scene_data.values()
            if 'prec' in img_data
        ]

        if not all_prec_values:
            self._logger.warning("No Precision results found to report.")
            return

        mean_total_ms = np.mean(all_ms_values)
        mean_total_prec = np.mean(all_prec_values)

        self._logger.info(f"--- {task_name.upper()} ---")
        self._logger.info(f"Mean MS: {mean_total_ms:.4f}, Mean Prec: {mean_total_prec:.4f}")


class HomographyAUCTask(HPatchesTask):
    _HOMOGRAPHY_METHODS = {
        "ransac": cv.RANSAC,
        "magsac": cv.USAC_MAGSAC,
        "lmeds": cv.LMEDS,
        "rho": cv.RHO
    }

    def __init__(self, logger, config):
        super().__init__(logger)
        self._eval_threshold = config.pop('eval_threshold', 5.0)
        self._homography_threshold = config.pop('homography_threshold', 3.0)
        self._homography_method = config.pop('homography_method', "ransac")

    def eval_task(self, matching_data, split):
        self._logger.info('Evaluating Full Pipeline (mAP) via Homography')
        results = {seq: {} for seq in split}

        for seq in split:
            if seq not in matching_data:
                continue

            for i in self._img_indices:
                data = matching_data[seq].get(i)
                if not data or not data['matches']:
                    continue

                kp_ref = data['kp_ref']
                kp_tgt = data['kp_tgt']
                matches = data['matches']

                pts_ref = np.array([kp_ref[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
                pts_tgt_pred = np.array([kp_tgt[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)

                H_gt = data['H']
                H_pred, mask = cv.findHomography(pts_ref, pts_tgt_pred,
                                                 self._HOMOGRAPHY_METHODS[self._homography_method],
                                                 self._homography_threshold)

                if H_pred is None:
                    continue

                h, w = data['ref_shape'][:2]
                corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

                corners_gt = cv.perspectiveTransform(corners, H_gt)
                corners_pred = cv.perspectiveTransform(corners, H_pred)

                error = np.mean(np.linalg.norm(corners_gt - corners_pred, axis=2))
                results[seq][i] = {'error': error}

        return results

    def report_metrics(self, results, task_name="Homography AUC"):
        all_errors = [
            img['error']
            for s in results.values()
            for img in s.values()
            if 'error' in img
        ]

        if not all_errors:
            self._logger.warning("No results found to report.")
            return

        thresholds = np.linspace(0, self._eval_threshold, 100)
        acc_curve = [np.mean(np.array(all_errors) < t) for t in thresholds]

        global_auc = np.trapezoid(acc_curve, thresholds) / self._eval_threshold
        self._logger.info(f"--- {task_name.upper()} ---")
        self._logger.info(f"Mean AUC@{self._eval_threshold}px: {global_auc:.4f}")
