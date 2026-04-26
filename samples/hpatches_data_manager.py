import numpy as np
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

            ref_path = scene_dir / "1.ppm"
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
                        'H': H
                    }

            dataset[scene] = scene_data

        return dataset
