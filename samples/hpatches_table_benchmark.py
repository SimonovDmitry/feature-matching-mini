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


def parse_metrics_log(log_output, task_names):
    results = {}

    patterns = {
        'matchingap': {
            'mean_ap': r'Mean Total AP:\s*([\d\.]+)',
        },
        'matchingscore': {
            'mean_ms': r'Mean MS:\s*([\d\.]+)',
            'mean_prec': r'Mean Prec:\s*([\d\.]+)',
        },
        'homographyauc': {
            'mean_auc': r'Mean AUC@[\d\.]+px:\s*([\d\.]+)',
        }
    }


    for task_name in task_names:
        task_patterns = patterns.get(task_name.lower(), {})
        for key, pattern in task_patterns.items():
            match = re.search(pattern, log_output)
            if match:
                results[f'{task_name}_{key}'] = float(match.group(1))

    return results


def run_benchmark(detector, descriptor, matcher, dataset_path, tasks, device='cpu', num_scenes=None):
    all_results = {}

    for task in tasks:
        cmd = [
            sys.executable, '-m', 'samples.hpatches_benchmark',
            '-det', detector,
            '-des', descriptor,
            '-mat', matcher,
            '-t', task,
            '-p', str(dataset_path),
        ]

        if device:
            cmd.extend(['-d', device])

        if num_scenes:
            cmd.extend(['-n', str(num_scenes)])

        logger.info(f"Running {task}: {detector}+{descriptor}+{matcher}")

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
                logger.warning(f"Failed {task}: {detector}+{descriptor}+{matcher}")
                logger.debug(f"Error output: {stderr}")
                return False, {}

            task_results = parse_metrics_log(output, [task])
            all_results.update(task_results)

        except Exception as e:
            logger.warning(f"Error running {task}: {e}")
            return False, {}
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()

    return True, all_results


def get_all_combinations():
    combinations = []

    for detector in DETECTOR_DESCRIPTOR_COMPATIBILITY:
        descriptors = DETECTOR_DESCRIPTOR_COMPATIBILITY.get(detector, [])

        for descriptor in descriptors:
            matchers = DESCRIPTOR_MATCHER_COMPATIBILITY.get(descriptor, [])

            for matcher in matchers:
                combinations.append((detector, descriptor, matcher))

    return combinations


def table_benchmark(dataset_path, output_csv, tasks, device='cpu', num_scenes=None):
    combinations = get_all_combinations()
    logger.info(f"Found {len(combinations)} valid combinations")
    logger.info(f"Tasks to run: {tasks}")

    all_results = []

    for detector, descriptor, matcher in combinations:
        combo_result = {
            'detector': detector,
            'descriptor': descriptor,
            'matcher': matcher,
            'device': device,
            'num_scenes': num_scenes if num_scenes else 'all',
        }

        success, metrics = run_benchmark(detector, descriptor, matcher, dataset_path, tasks, device, num_scenes)
        if not success:
            logger.warning(f"Skipping: {detector}+{descriptor}+{matcher}")
            continue

        combo_result.update(metrics)
        all_results.append(combo_result)

        logger.info(f"Completed: {detector}+{descriptor}+{matcher}")

    save_results_to_csv(all_results, output_csv, tasks)


def save_results_to_csv(results, output_path, tasks):
    if not results:
        logger.warning("No results to save")
        return

    df = pd.DataFrame(results)
    base_columns = ['detector', 'descriptor', 'matcher', 'device', 'num_scenes']

    metric_columns = []
    for task in tasks:
        if task.lower() == 'matchingap':
            metric_columns.append('matchingap_mean_ap')
        elif task.lower() == 'matchingscore':
            metric_columns.extend(['matchingscore_mean_ms', 'matchingscore_mean_prec'])
        elif task.lower() == 'homographyauc':
            metric_columns.append('homographyauc_mean_auc')

    columns_order = base_columns + metric_columns
    existing_columns = [col for col in columns_order if col in df.columns]
    df = df[existing_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Columns: {list(df.columns)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HPatches benchmarks for all combinations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    available_devices = ['cpu', 'cuda', 'mps']
    available_tasks = ['matchingap', 'matchingscore', 'homographyauc']

    parser.add_argument('-p', '--path', type=Path, required=True,
                        help='Path to hpatches-sequences-release folder')
    parser.add_argument('-o', '--output', type=Path, default=Path('hpatches_results.csv'),
                        help='Output CSV file path')
    parser.add_argument('-t', '--tasks', type=str, nargs='+', choices=available_tasks,
                        default=['matchingap', 'matchingscore', 'homographyauc'], help='Tasks to run (default: all)')
    parser.add_argument('-d', '--device', type=str, default=None, choices=available_devices,
                        help='Device to run on')
    parser.add_argument('-n', '--num-scenes', type=int, default=116,
                        help='Number of scenes to process (default: all 116)')

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
        device=args.device,
        num_scenes=args.num_scenes,
    )

    logger.info("Benchmark completed successfully")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
