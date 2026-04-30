import cv2 as cv
import numpy as np
import pandas as pd
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
        self.pixel_threshold = config.get('pixel_threshold', 5.0)

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

        distances = np.linalg.norm(pts_tgt_pred - pts_tgt_gt, axis=2).flatten()
        labels = (distances <= self.pixel_threshold).astype(int)

        return {
            'labels': labels,
            'distances': distances,
            'num_kp_ref': len(kp_ref),
            'num_matches': len(matches),
            'scores': np.array([-m.distance for m in matches])
        }


class MatchingAPTask(BaseMatchingTask):
    def eval_task(self, matching_data, split):
        self._logger.info(f'Evaluating Feature Matching (mAP) @ {self.pixel_threshold}px')
        results = {seq: {} for seq in split}

        for seq in split:
            if seq not in matching_data: continue
            for i in self._img_indices:
                data = matching_data[seq].get(i)
                res = self._get_match_results(data) if data else None

                if res is not None:
                    _, _, ap = self._pr(res['scores'], res['labels'], numpos=res['num_kp_ref'])
                    results[seq][i] = {'ap': ap}
        return results

    def report_metrics(self, results, task_name="Feature Matching (mAP)"):
        rows = [{'Scene': s, 'mAP': np.mean([m['ap'] for m in r.values()])}
                for s, r in results.items() if r]
        df = pd.DataFrame(rows).sort_values('mAP', ascending=False)
        self._logger.info(f"\n--- {task_name.upper()} ---\n{df.to_string(index=False)}")
        self._logger.info(f"Mean Total AP: {df['mAP'].mean():.4f}")


class MatchingScoreTask(BaseMatchingTask):
    def eval_task(self, matching_data, split):
        self._logger.info(f'Evaluating Matching Score & Precision @ {self.pixel_threshold}px')
        results = {seq: {} for seq in split}

        for seq in split:
            if seq not in matching_data: continue
            for i in self._img_indices:
                data = matching_data[seq].get(i)
                res = self._get_match_results(data) if data else None

                if res is not None:
                    num_inliers = np.sum(res['labels'])
                    results[seq][i] = {
                        'ms': num_inliers / res['num_kp_ref'],
                        'prec': num_inliers / res['num_matches'] if res['num_matches'] > 0 else 0
                    }
        return results

    def report_metrics(self, results, task_name="Matching Score & Precision"):
        rows = []
        for seq, res in results.items():
            if not res: continue
            rows.append({
                'Scene': seq,
                'MS': np.mean([m['ms'] for m in res.values()]),
                'Prec': np.mean([m['prec'] for m in res.values()])
            })

        df = pd.DataFrame(rows).sort_values('MS', ascending=False)
        self._logger.info(f"\n--- {task_name.upper()} ---\n{df.to_string(index=False)}")
        self._logger.info(f"Mean MS: {df['MS'].mean():.4f}, Mean Prec: {df['Prec'].mean():.4f}")


class HomographyAUCTask(HPatchesTask):
    _HOMOGRAPHY_METHODS = {
        "ransac": cv.RANSAC,
        "magsac": cv.USAC_MAGSAC,
        "lmeds": cv.LMEDS,
        "rho": cv.RHO
    }

    def __init__(self, logger, config):
        super().__init__(logger)
        self._pixel_threshold = config.pop('pixel_threshold', 5.0)
        self._homography_method = config.pop('homography_method', "ransac")


    def eval_task(self, matching_data, split):
        self._logger.info('Evaluating Full Pipeline (mAP) via Homography')
        results = {seq: {} for seq in split}

        for seq in split:
            if seq not in matching_data:
                continue

            scene_errors = []
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
                                                 self._pixel_threshold)

                if H_pred is None:
                    continue

                h, w = data['ref_shape'][:2]
                corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

                corners_gt = cv.perspectiveTransform(corners, H_gt)
                corners_pred = cv.perspectiveTransform(corners, H_pred)

                error = np.mean(np.linalg.norm(corners_gt - corners_pred, axis=2))
                scene_errors.append(error)

            scene_errors = np.array(scene_errors)

            thresholds = np.linspace(0, self._pixel_threshold, 100)
            accuracies = [np.mean(scene_errors < t) for t in thresholds]

            area = np.trapezoid(accuracies, thresholds)
            scene_auc = area / self._pixel_threshold

            results[seq] = {'auc': scene_auc}

        return results

    def report_metrics(self, results, task_name="Homography Estimation"):
        rows = []
        all_scene_aucs = []

        for scene, data in results.items():
            if 'auc' in data:
                rows.append({
                    'Scene': scene,
                    f'AUC@{self._pixel_threshold}px': data['auc']
                })
                all_scene_aucs.append(data['auc'])

        if not rows:
            self._logger.info("No Accuracy results to report")
            return

        df = pd.DataFrame(rows).sort_values(f'AUC@{self._pixel_threshold}px', ascending=False)

        self._logger.info(f"--- {task_name.upper()} RESULTS ---")
        self._logger.info(df.to_string(index=False))

        mean_auc = np.mean(all_scene_aucs)
        self._logger.info(f"Mean AUC over all scenes: {mean_auc:.2f} (Threshold: {self._pixel_threshold}px)")
