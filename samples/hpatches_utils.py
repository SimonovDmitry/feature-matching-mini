import cv2 as cv
import numpy as np
from scipy import spatial
import pandas as pd
from collections import defaultdict
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm

try:
    import torch
    TORCH_GPU_IS_AVAILABLE = torch.cuda.is_available()

except ImportError:
    TORCH_GPU_IS_AVAILABLE = False


class HPatchesDataManager:
    _img_types = ['ref', 'e1', 'e2', 'e3', 'e4', 'e5', 'h1', 'h2', 'h3', 'h4', 'h5']
    _patch_size = 65

    def __init__(self, raw_data_path, logger):
        self._raw_data_path = Path(raw_data_path)
        self._logger = logger

    def load_patches_from_image(self, scene_name, img_type):
        img_path = self._raw_data_path / scene_name / f"{img_type}.png"
        if not img_path.exists():
            return None

        strip = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        if strip is None: return None

        h, w = strip.shape
        n_patches = h // self._patch_size

        patches = strip.reshape(n_patches, self._patch_size, self._patch_size)
        return patches

    def load_dataset(self, num_scenes=None):
        dataset = {}
        all_scenes = [d.name for d in self._raw_data_path.iterdir() if d.is_dir()]
        all_scenes.sort()

        if num_scenes is not None and num_scenes > 0:
            scenes = all_scenes[:num_scenes]
        else:
            scenes = all_scenes

        self._logger.info(f"Loading {len(scenes)} scenes")
        for scene in scenes:
            scene_data = {}
            self._logger.info(f"name file {scene}")
            for t in self._img_types:
                patches = self.load_patches_from_image(scene, t)
                if patches is not None:
                    scene_data[t] = patches

            dataset[scene] = scene_data

        return dataset


class HPatchesTask(ABC):
    _TASKS = {}
    _tp = ['e', 'h', 't']
    _id2t = {0: {'e': 'ref', 'h': 'ref', 't': 'ref'},
             1: {'e': 'e1', 'h': 'h1', 't': 't1'},
             2: {'e': 'e2', 'h': 'h2', 't': 't2'},
             3: {'e': 'e3', 'h': 'h3', 't': 't3'},
             4: {'e': 'e4', 'h': 'h4', 't': 't4'},
             5: {'e': 'e5', 'h': 'h5', 't': 't5'}}

    def __init__(self, logger):
        self._logger = logger

    def __init_subclass__(cls, register=True, **kwargs):
        super().__init_subclass__(**kwargs)

        if register:
            key = cls.__name__.replace("Task", "").lower()
            if key:
                HPatchesTask._TASKS[key] = cls

    @classmethod
    def create(cls, task_name, logger):
        task_class = cls._TASKS.get(task_name.lower())
        if not task_class:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(cls._TASKS.keys())}")
        return task_class(logger)

    @abstractmethod
    def eval_task(self, descriptors, split):
        pass

    def report_metrics(self, results, task_name=""):
        rows = []
        for scene, res in results.items():
            aps = [m['ap'] for t in res.values() for m in t.values() if 'ap' in m]
            if aps:
                rows.append({'Scene': scene, 'mAP': np.mean(aps)})

        if not rows:
            self._logger.info("No results to display.")
            return

        df = pd.DataFrame(rows).sort_values('mAP', ascending=False)
        self._logger.info(f" {task_name.upper()} RESULTS ")
        self._logger.info(df.to_string(index=False))
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

    @classmethod
    def _roc(cls, scores, labels, numpos=None):
        [tp, fp, p, n, perm] = cls._tpfp(scores, labels, numpos)

        small = 1e-10
        tpr = tp / float(np.maximum(p, small))
        fpr = fp / float(np.maximum(n, small))
        return fpr, tpr, np.trapezoid(tpr, fpr)

    @classmethod
    def _seqs_lengths(cls, seqs):
        N = {}
        for seq in seqs:
            N[seq] = seqs[seq].N
        return N

    @classmethod
    def _dist_matrix(cls, D1, D2, distance):
        if distance == 'L2':
            if TORCH_GPU_IS_AVAILABLE:
                import torch
                with torch.no_grad():

                    D = torch.cdist(torch.from_numpy(D1).cuda().float(),
                                    torch.from_numpy(D2).cuda().float(),
                                    p=2).detach().cpu().numpy()
            else:
                D = spatial.distance.cdist(D1, D2, 'euclidean')
        elif distance == 'L1':
            if TORCH_GPU_IS_AVAILABLE:
                import torch
                with torch.no_grad():
                    D = torch.cdist(torch.from_numpy(D1).cuda(),
                                    torch.from_numpy(D2).cuda(),
                                    p=1).detach().cpu().numpy()
            else:
                D = spatial.distance.cdist(D1, D2, 'cityblock')
        elif distance == 'masked_L1':
            [desc1, masks1] = np.split(D1, 2, axis=1)
            [desc2, masks2] = np.split(D2, 2, axis=1)
            D = spatial.distance.cdist(desc1 * masks1, desc2 * masks2, 'cityblock')
        else:
            raise ValueError('Unknown distance - valid options are |L2|L1|masked_L1|')
        return D


class MatchingTask(HPatchesTask):
    def __init__(self, logger):
        super().__init__(logger)

    def eval_task(self, descr, split):
        self._logger.info(f'Evaluating task matching')

        results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        pbar = tqdm(split['test'])
        for seq in pbar:
            d_ref = descr[seq].get('ref')
            if d_ref is None:
                continue

            gt_l = np.arange(d_ref.shape[0])
            for t in self._tp:
                for i in range(1, 6):
                    key = t + str(i)

                    if key not in descr[seq]:
                        continue

                    d = descr[seq][key]
                    D = self._dist_matrix(d_ref, d, descr['distance'])
                    idx = np.argmin(D, axis=1)
                    m_l = np.equal(idx, gt_l)
                    results[seq][t][i]['sr'] = np.count_nonzero(m_l) / float(m_l.shape[0])
                    m_d = D[gt_l, idx]
                    pr, rc, ap = self._pr(-m_d, m_l, numpos=m_l.shape[0])
                    results[seq][t][i]['ap'] = ap
                    results[seq][t][i]['pr'] = pr
                    results[seq][t][i]['rc'] = rc

        self._logger.info(f"Matching task finished")
        return results


