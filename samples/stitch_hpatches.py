import argparse
import logging
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))  # noqa: E402

from samples.hpatches_data_manager import HPatchesDataManager  # noqa: E402
from samples.utils import (build_hpatches_benchmark_config, build_hpatches_feature_matcher_config) # noqa: E402

from src.feature_matcher import FeatureMatcherCV2  # noqa: E402
from src.matchers import OpenCVMatcher  # noqa: E402

logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger("HPatchesStitching")

HOMOGRAPHY_METHODS = {
    "ransac": cv.RANSAC,
    "magsac": cv.USAC_MAGSAC,
    "lmeds": cv.LMEDS,
    "rho": cv.RHO
}


def stitch_pair(img_ref, img_tgt, H):
    if H is None or H.shape != (3, 3):
        error_img = np.zeros((300, 600, 3), dtype=np.uint8)
        cv.putText(error_img, "Homography Failed (None)", (30, 160), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return error_img

    h1, w1 = img_ref.shape[:2]
    h2, w2 = img_tgt.shape[:2]

    corners_ref = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners_tgt = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 2)

    try:
        corners_ref_trans = cv.perspectiveTransform(corners_ref, H).reshape(-1, 2)
    except cv.error:
        error_img = np.zeros((300, 600, 3), dtype=np.uint8)
        cv.putText(error_img, "Invalid Homography Matrix", (30, 160),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return error_img

    all_corners = np.vstack((corners_ref_trans, corners_tgt))
    x_min, y_min = np.int32(all_corners.min(axis=0) - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0) + 0.5)

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    if canvas_w > 8000 or canvas_h > 8000 or canvas_w <= 0 or canvas_h <= 0:
        error_img = np.zeros((300, 600, 3), dtype=np.uint8)
        cv.putText(error_img, "Degenerate Canvas Size", (30, 160),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return error_img

    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)

    H_ref_to_canvas = translation @ H
    warped_ref = cv.warpPerspective(img_ref, H_ref_to_canvas, (canvas_w, canvas_h))
    warped_tgt = cv.warpPerspective(img_tgt, translation, (canvas_w, canvas_h))

    mask_ref = (warped_ref > 0).astype(np.uint8)
    mask_tgt = (warped_tgt > 0).astype(np.uint8)
    overlap = (mask_ref & mask_tgt)
    non_overlap = ~overlap

    stitched = (warped_ref * (mask_ref & non_overlap)) + (warped_tgt * (mask_tgt & non_overlap))
    overlap_blend = cv.addWeighted(warped_ref, 0.5, warped_tgt, 0.5, 0)
    stitched += overlap_blend * overlap

    return stitched


def run_stitching(cli_args):
    base_config = build_hpatches_benchmark_config(cli_args)
    fm_config = build_hpatches_feature_matcher_config(cli_args)

    det = cli_args.detector
    desc = cli_args.descriptor
    mat = cli_args.matcher

    combo_name = f"{det}+{desc}+{mat}"
    output_dir = cli_args.output_dir / f"result_stitching_{combo_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running stitching for combination: {combo_name}")
    logger.info(f"Saving results to: {output_dir.resolve()}")

    feature_matcher = FeatureMatcherCV2(detector=det, descriptor=desc, matcher=mat, logger=logger, config=fm_config)
    dm = HPatchesDataManager(logger=logger, config=base_config['dataset'])

    h_method_flag = HOMOGRAPHY_METHODS.get(cli_args.homography_method.lower(), cv.RANSAC)
    total_processed = 0

    try:
        while dm.has_more_data():
            current_batch = dm.load_batch()
            if not current_batch:
                break

            for scene_name, scene_data in current_batch.items():
                if cli_args.viewpoint_only and not scene_name.startswith('v_'):
                    logger.info(f"Skipping non-viewpoint scene: {scene_name}")
                    continue

                img_ref = scene_data['ref_img']
                scene_dir = output_dir / scene_name

                for i, target in scene_data['targets'].items():
                    img_tgt = target['image']
                    H_gt = target['H']

                    features_ref, features_tgt, correspondences = feature_matcher.match(img_ref, img_tgt)
                    matches = correspondences['matches']

                    H_pred = None
                    if len(matches) >= 4:
                        pts_ref = np.float32([features_ref['kp'][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                        pts_tgt = np.float32([features_tgt['kp'][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                        H_pred, _ = cv.findHomography(pts_ref, pts_tgt, h_method_flag, cli_args.homography_threshold)

                    stitched_pred = stitch_pair(img_ref, img_tgt, H_pred)
                    stitched_gt = stitch_pair(img_ref, img_tgt, H_gt)

                    pair_dir = scene_dir / str(i)
                    pair_dir.mkdir(parents=True, exist_ok=True)

                    cv.imwrite(str(pair_dir / "stitched_pred.jpg"), stitched_pred)
                    cv.imwrite(str(pair_dir / "stitched_gt.jpg"), stitched_gt)

                    total_processed += 1

        dm.reset()
        logger.info(f"Successfully processed and stitched {total_processed} pairs across scenes.")

    except Exception as e:
        logger.error(f"Error during stitching process: {e}", exc_info=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run batch-based HPatches stitching for a specific algorithm combination",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    available_devices = ['cpu', 'cuda', 'mps']
    available_matchers_modes = list(OpenCVMatcher._MODE)

    parser.add_argument('-p', '--path', type=Path, required=True,
                        help='Path to hpatches-sequences-release folder')
    parser.add_argument('-o', '--output-dir', type=Path, default=Path('.'),
                        help='Root directory for output folders')

    parser.add_argument('-det', '--detector', type=str, default='sift', help='Detector name')
    parser.add_argument('-des', '--descriptor', type=str, default='sift', help='Descriptor name')
    parser.add_argument('-mat', '--matcher', type=str, default='bf', help='Matcher name')

    parser.add_argument('-d', '--device', type=str, default=None,
                        choices=available_devices, help='Device')
    parser.add_argument('-n', '--num-scenes', type=int, default=116, help='Number of scenes to process')
    parser.add_argument('-sbs', '--scenes-batch-size', type=int, default=4,
                        help='Batch size for processing scenes')
    parser.add_argument('-mp', '--modelpath', type=Path, default=None, help='Path to models')
    parser.add_argument('-vo', '--viewpoint-only', action='store_true',
                        help='Process only viewpoint scenes starting with "v_"')

    task_group = parser.add_argument_group('Homography config')
    task_group.add_argument('-et', '--eval-thresholds', type=float, nargs='+', default=[5.0],
                            help='Pixel thresholds (1.0 3.0 5.0 10.0)')
    task_group.add_argument('-hm', '--homography-method', type=str, default='ransac',
                            choices=list(HOMOGRAPHY_METHODS.keys()), help='Homography estimation method')
    task_group.add_argument('-ht', '--homography-threshold', type=float, default=3.0,
                            help='RANSAC reprojection threshold')

    det_group = parser.add_argument_group('Detector config')
    det_group.add_argument('-dn', '--det-nfeatures', type=int, default=None,
                           help='Max features for detector')
    det_group.add_argument('-do', '--det-noctave', type=int, default=None, help='Octave layers')
    det_group.add_argument('-dt', '--det-threshold', type=float, default=None, help='Detection threshold')

    des_group = parser.add_argument_group('Descriptor config')
    des_group.add_argument('-dsen', '--des-nfeatures', type=int, default=None,
                           help='Max features for descriptor')
    des_group.add_argument('-dsdt', '--des-threshold', type=float, default=None,
                           help='Descriptor threshold')
    des_group.add_argument('-dss', '--des-scale', type=float, default=None, help='Scale factor')

    mat_group = parser.add_argument_group('Matcher config')
    mat_group.add_argument('-mat_m', '--matcher_mode', type=str, default='simple',
                           choices=available_matchers_modes, help='Matching mode')
    mat_group.add_argument('-mr', '--mat-ratio', type=float, default=None, help='Ratio threshold for KNN')
    mat_group.add_argument('-mc', '--mat-cross-check', action='store_true', default=None,
                           help='Enable cross-check')

    return parser.parse_args()


def main():
    args = parse_args()
    if not args.path.exists():
        logger.error(f"Dataset path does not exist: {args.path}")
        return 1

    run_stitching(args)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)