
NEURAL_DETECTORS = {'superpoint', 'superpoint_lightglue', 'disk_lightglue', 'sift_lightglue', 'aliked_lightglue',
                    'doghardnet_lightglue'}

NEURAL_DESCRIPTORS = {'superpoint', 'superpoint_lightglue', 'disk_lightglue', 'sift_lightglue', 'aliked_lightglue',
                      'doghardnet_lightglue'}

NEURAL_MATCHERS = {'lightglue'}

OPENCV_DETECTORS = {'sift', 'orb', 'fast', 'akaze', 'brisk', 'kaze', 'gftt', 'mser', 'agast', 'blob', 'star',
                    'harrislaplace', 'msd'}

OPENCV_DESCRIPTORS = {'sift', 'orb', 'akaze', 'brisk', 'kaze', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid',
                      'vgg', 'boostdesc'}

OPENCV_MATCHERS = {'bf', 'flann'}
OPENCV_MATCHERS_MODE = {'simple', 'knn'}


NEURAL_ALGORITHMS = NEURAL_DETECTORS | NEURAL_DESCRIPTORS | NEURAL_MATCHERS
OPENCV_ALGORITHMS = OPENCV_DETECTORS | OPENCV_DESCRIPTORS | OPENCV_MATCHERS
ALL_ALGORITHMS = NEURAL_ALGORITHMS | OPENCV_ALGORITHMS

DETECTOR_DESCRIPTOR_COMPATIBILITY = {
    'sift': ['sift', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
    'orb': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
    'fast': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
    'akaze': ['sift', 'orb', 'akaze', 'brisk', 'kaze', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid',
              'vgg', 'boostdesc'],
    'brisk': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
    'kaze': ['sift', 'orb', 'brisk', 'kaze', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg',
             'boostdesc'],
    'gftt': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
    'mser': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
    'agast': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
    'blob': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
    'star': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],
    'harrislaplace': ['sift', 'orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg',
                      'boostdesc'],
    'msd': ['orb', 'brisk', 'brief', 'freak', 'daisy', 'latch', 'beblid', 'teblid', 'vgg', 'boostdesc'],

    'superpoint': ['superpoint'],
    'superpoint_lightglue': ['superpoint_lightglue'],
    'disk_lightglue': ['disk_lightglue'],
    'sift_lightglue': ['sift_lightglue'],
    'aliked_lightglue': ['aliked_lightglue'],
    'doghardnet_lightglue': ['doghardnet_lightglue'],
}

DESCRIPTOR_MATCHER_COMPATIBILITY = {
    'sift': ['bf', 'flann'],
    'orb': ['bf', 'flann'],
    'akaze': ['bf', 'flann'],
    'brisk': ['bf', 'flann'],
    'kaze': ['bf', 'flann'],
    'brief': ['bf', 'flann'],
    'freak': ['bf', 'flann'],
    'daisy': ['bf', 'flann'],
    'latch': ['bf', 'flann'],
    'beblid': ['bf', 'flann'],
    'teblid': ['bf', 'flann'],
    'vgg': ['bf', 'flann'],
    'boostdesc': ['bf', 'flann'],

    'superpoint': ['bf', 'flann'],
    'superpoint_lightglue': ['lightglue'],
    'disk_lightglue': ['lightglue'],
    'sift_lightglue': ['lightglue'],
    'aliked_lightglue': ['lightglue'],
    'doghardnet_lightglue': ['lightglue'],
}
