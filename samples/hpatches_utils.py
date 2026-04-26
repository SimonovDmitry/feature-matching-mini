import cv2 as cv
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path

from src.image_utils import read_image


class HPatchesDataManager:
    def __init__(self, raw_data_path, logger):
        self._raw_data_path = Path(raw_data_path)
        self._logger = logger
        self._img_indices = [2, 3, 4, 5, 6]

    def load_dataset(self, num_scenes=None):
        dataset = {}
        all_scenes = [d.name for d in self._raw_data_path.iterdir() if d.is_dir()]
        all_scenes.sort()

        if num_scenes is not None and num_scenes > 0:
            scenes = all_scenes[:num_scenes]
        else:
            scenes = all_scenes

        self._logger.info(f"Loading {len(scenes)} scenes for Full Image Matching")

        for scene in scenes:
            scene_dir = self._raw_data_path / scene
            self._logger.info(f"Loading {scene_dir}")

            ref_path = scene_dir / f"1.ppm"
            img_ref = read_image(ref_path)

            if img_ref is None:
                self._logger.warning(f"Could not read reference image in {scene}")
                continue

            scene_data = {'ref_img': img_ref, 'targets': {}}

            for i in self._img_indices:
                target_path = scene_dir / f"{i}.ppm"
                img_target = read_image(target_path)

                h_path = scene_dir / f"H_1_{i}"
                if img_target is not None and h_path.exists():
                    H = np.loadtxt(str(h_path))
                    scene_data['targets'][i] = {
                        'image': img_target,
                        f'H': H
                    }

            dataset[scene] = scene_data

        return dataset


class HPatchesTask(ABC):
    _TASKS = {}

    def __init__(self, logger):
        self._logger = logger

    def __init_subclass__(cls, register=True, **kwargs):
        super().__init_subclass__(**kwargs)

        if register:
            key = cls.__name__.replace("Task", "").lower()
            if key:
                HPatchesTask._TASKS[key] = cls

    @classmethod
    def create(cls, task_name, logger, pixel_threshold=5.0):
        task_class = cls._TASKS.get(task_name.lower())
        if not task_class:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(cls._TASKS.keys())}")
        return task_class(logger, pixel_threshold)

    @abstractmethod
    def eval_task(self, descriptors, split):
        pass

    def report_metrics(self, results, task_name=""):
        rows = []

        for scene, res in results.items():
            aps = [metrics['ap'] for metrics in res.values() if 'ap' in metrics]
            if aps:
                rows.append({'Scene': scene, 'mAP': np.mean(aps)})

        if not rows:
            self._logger.info("No results to display.")
            return

        df = pd.DataFrame(rows).sort_values('mAP', ascending=False)
        self._logger.info(f"--- {task_name.upper()} RESULTS ---")
        self._logger.info("\n" + df.to_string(index=False))
        self._logger.info(f"Mean Total AP: {df['mAP'].mean():.4f}")

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


class MatchingTask(HPatchesTask):
    def __init__(self, logger, pixel_threshold=5.0):
        super().__init__(logger)
        self.pixel_threshold = pixel_threshold

    def eval_task(self, matching_data, split):
        self._logger.info(f'Evaluating Full Pipeline (mAP) via Homography')
        results = {seq: {} for seq in split}

        for seq in split:
            if seq not in matching_data: continue

            for i in range(2, 7):
                data = matching_data[seq].get(i)
                if not data or not data['matches']:
                    continue

                kp_ref = data['kp_ref']
                kp_tgt = data['kp_tgt']
                matches = data['matches']
                H = data['H']

                pts_ref = np.array([kp_ref[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
                pts_tgt_pred = np.array([kp_tgt[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
                pts_tgt_gt = cv.perspectiveTransform(pts_ref, H)

                distances = np.linalg.norm(pts_tgt_pred - pts_tgt_gt, axis=2).flatten()
                labels = (distances <= self.pixel_threshold).astype(int)

                scores = np.array([-m.distance for m in matches])
                precision, recall, ap = self._pr(scores, labels, numpos=len(kp_ref))

                results[seq][i] = {
                    'ap': ap,
                    'precision': precision[-1] if len(precision) > 0 else 0,
                    'recall': recall[-1] if len(recall) > 0 else 0
                }

        return results