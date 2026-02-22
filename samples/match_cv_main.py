import argparse
import sys
from match_cv_scripts import FeatureMatcherCV2, logger

def parser():
    arg_parser = argparse.ArgumentParser(
        description="Matching points in two images using OpenCV algorithms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-met', '--method', type=str, default='sift',
        help='Algorithm for detecting and creating descriptors of key points')
    arg_parser.add_argument('-mat' ,'--matcher', type=str, default='bf',
        help='Matching algorithm')

    arg_parser.add_argument('-i1', '--image1', type=str, required=True,
        help='Path to the first image')
    arg_parser.add_argument('-i2', '--image2', type=str, required=True,
        help='Path to the second image')
    arg_parser.add_argument('-s', '--save', action='store_true',
        default = False, help='Save the result in JPG format')

    return arg_parser.parse_args()


def main():
    args = parser()

    try:
        logger.info(f"Launching opencv_sample")
        matcher = FeatureMatcherCV2(args.method)
        if not matcher.load_images(args.image1, args.image2):
            return 1

        matcher.run_matching(args.matcher)
        matcher.visualize_matching(save = args.save)
        logger.info("The opencv_sample program has completed successfully")
        return 0

    except Exception as e:
        logger.exception(f"An error occurred during execution: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main() or 0)