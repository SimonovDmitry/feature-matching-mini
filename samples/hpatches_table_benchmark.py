import argparse
import re
import subprocess
import sys
import logging
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # noqa: E402

from src.algorithms import DETECTOR_DESCRIPTOR_COMPATIBILITY, DESCRIPTOR_MATCHER_COMPATIBILITY  # noqa: E402

logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger("HPatchesTableBenchmark")


def parse_metrics_log(log_output, task_names, thresholds):
    results = {}

    for threshold in thresholds:
        patterns = {
            'matchingap': {
                f'matchingap_mean_ap_{threshold}': (
                    rf'--- END-TO-END PIPELINE \[MATCHINGAP\] @ {threshold}(?:\.0)?px ---'
                    rf'[\s\S]*?Mean Total AP:\s*([\d\.]+)'
                ),
            },
            'matchingscore': {
                f'matchingscore_mean_ms_{threshold}': (
                    rf'--- END-TO-END PIPELINE \[MATCHINGSCORE\] @ {threshold}(?:\.0)?px ---'
                    rf'[\s\S]*?Mean MS:\s*([\d\.]+)'
                ),
                f'matchingscore_mean_prec_{threshold}': (
                    rf'--- END-TO-END PIPELINE \[MATCHINGSCORE\] @ {threshold}(?:\.0)?px ---'
                    rf'[\s\S]*?Mean Prec:\s*([\d\.]+)'
                ),
            },
            'homographyauc': {
                f'homographyauc_mean_auc_{threshold}': (
                    rf'--- END-TO-END PIPELINE \[HOMOGRAPHYAUC\] @ {threshold}(?:\.0)?px ---'
                    rf'[\s\S]*?Mean AUC:\s*([\d\.]+)'
                ),
            }
        }

        for task_name in task_names:
            task_patterns = patterns.get(task_name.lower(), {})
            for key, pattern in task_patterns.items():
                match = re.search(pattern, log_output, re.IGNORECASE)
                if match:
                    results[key] = float(match.group(1))
                else:
                    logger.debug(f"No match for: {key}")

    return results


def run_benchmark(detector, descriptor, matcher, dataset_path, tasks, thresholds, device='cpu', num_scenes=None):
    cmd = [
        sys.executable, '-m', 'samples.hpatches_benchmark',
        '-det', detector,
        '-des', descriptor,
        '-mat', matcher,
        '-t'] + tasks + [
        '-p', str(dataset_path),
        '-et'] + [str(t) for t in thresholds]

    if device:
        cmd.extend(['-d', device])

    if num_scenes:
        cmd.extend(['-n', str(num_scenes)])

    logger.info(f"Running: {detector}+{descriptor}+{matcher}")

    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        output = stdout + stderr

        if process.returncode != 0:
            logger.warning(f"Failed: {detector}+{descriptor}+{matcher}")
            logger.debug(f"Error output: {stderr}")
            return False, {}

        all_results = parse_metrics_log(output, tasks, thresholds)
        return True, all_results

    except Exception as e:
        logger.warning(f"Error: {e}")
        return False, {}
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()


def get_all_combinations():
    combinations = []

    for detector in DETECTOR_DESCRIPTOR_COMPATIBILITY:
        descriptors = DETECTOR_DESCRIPTOR_COMPATIBILITY.get(detector, [])

        for descriptor in descriptors:
            matchers = DESCRIPTOR_MATCHER_COMPATIBILITY.get(descriptor, [])

            for matcher in matchers:
                combinations.append((detector, descriptor, matcher))

    return combinations


def load_existing_results(output_path):
    if not output_path.exists():
        logger.info(f"No existing results found at {output_path}")
        return None, set()

    try:
        df = pd.read_csv(output_path)
        logger.info(f"Loaded {len(df)} existing results from {output_path}")

        existing_combos = set()
        for _, row in df.iterrows():
            combo = (row['detector'], row['descriptor'], row['matcher'])
            existing_combos.add(combo)

        logger.info(f"Found {len(existing_combos)} unique combinations already computed")
        return df, existing_combos

    except Exception as e:
        logger.warning(f"Error loading existing results: {e}")
        return None, set()


def save_single_result(output_path, new_result, tasks, thresholds):
    base_columns = ['detector', 'descriptor', 'matcher', 'device', 'num_scenes']

    metric_columns = []
    for task in tasks:
        if task.lower() == 'matchingap':
            for threshold in thresholds:
                metric_columns.append(f'matchingap_mean_ap_{threshold}')
        elif task.lower() == 'matchingscore':
            for threshold in thresholds:
                metric_columns.append(f'matchingscore_mean_ms_{threshold}')
                metric_columns.append(f'matchingscore_mean_prec_{threshold}')
        elif task.lower() == 'homographyauc':
            for threshold in thresholds:
                metric_columns.append(f'homographyauc_mean_auc_{threshold}')

    columns_order = base_columns + metric_columns

    if output_path.exists():
        df = pd.read_csv(output_path)
    else:
        df = pd.DataFrame(columns=columns_order)

    new_df = pd.DataFrame([new_result])
    df = pd.concat([df, new_df], ignore_index=True)

    existing_columns = [col for col in columns_order if col in df.columns]
    df = df[existing_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def table_benchmark(dataset_path, output_csv, tasks, thresholds, device='cpu', num_scenes=None, skip_existing=True):
    combinations = get_all_combinations()
    logger.info(f"Found {len(combinations)} valid combinations")
    logger.info(f"Tasks to run: {tasks}")
    logger.info(f"Thresholds: {thresholds}")

    existing_df, existing_combos = load_existing_results(output_csv)

    if skip_existing and existing_combos:
        remaining_combos = [c for c in combinations if c not in existing_combos]
        logger.info(f"Skipping {len(existing_combos)} already computed combinations")
        logger.info(f"Remaining: {len(remaining_combos)} combinations")
        combinations = remaining_combos

    if not combinations:
        logger.info("All combinations already computed!")
        return

    for detector, descriptor, matcher in combinations:
        logger.info(f"Processing: {detector}+{descriptor}+{matcher}")

        combo_result = {
            'detector': detector,
            'descriptor': descriptor,
            'matcher': matcher,
            'device': device,
            'num_scenes': num_scenes if num_scenes else 116,
        }

        success, metrics = run_benchmark(
            detector, descriptor, matcher,
            dataset_path, tasks, thresholds, device, num_scenes
        )

        if not success:
            logger.warning(f"Failed: {detector}+{descriptor}+{matcher}")
            continue

        combo_result.update(metrics)

        try:
            save_single_result(output_csv, combo_result, tasks, thresholds)
            logger.info(f"Saved: {detector}+{descriptor}+{matcher}")
            logger.info(f"Metrics: {metrics}")
        except Exception as e:
            logger.error(f"Error saving result: {e}")
            continue

    logger.info(f"All combinations processed!")
    logger.info(f"Results saved to: {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HPatches benchmarks for all combinations with incremental saving",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    available_devices = ['cpu', 'cuda', 'mps']
    available_tasks = ['matchingap', 'matchingscore', 'homographyauc']

    parser.add_argument('-p', '--path', type=Path, required=True,
                        help='Path to hpatches-sequences-release folder')
    parser.add_argument('-o', '--output', type=Path, default=Path('hpatches_results.csv'),
                        help='Output CSV file path')

    parser.add_argument('-t', '--tasks', type=str, nargs='+', choices=available_tasks,
                        default=['matchingap', 'matchingscore', 'homographyauc'],
                        help='Tasks to run')
    parser.add_argument('-et', '--eval-thresholds', type=float, nargs='+', default=[5.0],
                        help='Pixel thresholds (1.0 3.0 5.0 10.0)')

    parser.add_argument('-d', '--device', type=str, default=None, choices=available_devices,
                        help='Device to run on')
    parser.add_argument('-n', '--num-scenes', type=int, default=116,
                        help='Number of scenes to process')
    parser.add_argument('--no-skip', action='store_true',
                        help='Recompute already existing combinations')

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.path.exists():
        logger.error(f"Dataset path does not exist: {args.path}")
        return 1

    logger.info("HPatches Benchmark Table Generator")

    table_benchmark(
        dataset_path=args.path,
        output_csv=args.output,
        tasks=args.tasks,
        thresholds=args.eval_thresholds,
        device=args.device,
        num_scenes=args.num_scenes,
        skip_existing=not args.no_skip,
    )

    logger.info("Benchmark completed successfully")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
