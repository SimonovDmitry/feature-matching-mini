import re
import subprocess
import sys
from pathlib import Path

import cv2 as cv
import numpy as np


def test_hpatches_quality_evaluation_cli(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    reference = cv.imread(str(repo_root / "test_data" / "box.png"))
    assert reference is not None
    height, width = reference.shape[:2]

    dataset = tmp_path / "hpatches"
    scene = dataset / "v_smoke"
    scene.mkdir(parents=True)
    assert cv.imwrite(str(scene / "1.ppm"), reference)

    # Generate each target and its reference-to-target ground truth together.
    for index in range(2, 7):
        homography = np.eye(3)
        homography[:2, 2] = (3 * index, -2 * index)
        target = cv.warpPerspective(reference, homography, (width, height))
        assert cv.imwrite(str(scene / f"{index}.ppm"), target)
        np.savetxt(scene / f"H_1_{index}", homography)

    result = subprocess.run(
        [
            sys.executable, str(repo_root / "samples" / "hpatches_benchmark.py"),
            "--path", str(dataset),
            "--detector", "sift", "--descriptor", "sift", "--matcher", "bf",
            "--tasks", "matchingap", "matchingscore", "homographyauc",
            "--eval-thresholds", "3",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout + result.stderr
    for metric in ("MAP", "MS", "Prec", "AUC"):
        score = re.search(rf"@ 3\.0px, [^\n]*\b{metric}: (\d+\.\d+)", output)
        assert score is not None, f"Missing {metric} score at 3 px:\n{output}"
        assert 0 < float(score.group(1)) <= 1, score.group(0)
