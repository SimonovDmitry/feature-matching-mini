from screeninfo import get_monitors
import cv2 as cv
import numpy as np
import torch


def tensor_to_opencv(tensor):
    img_numpy = tensor.detach().cpu().numpy()
    if img_numpy.ndim == 2:
        img_numpy = img_numpy[:, :, np.newaxis]
    if img_numpy.shape[0] in [1, 3]:
        img_numpy = np.transpose(img_numpy, (1, 2, 0))
    if img_numpy.max() <= 1.0:
        img_numpy = img_numpy * 255
    img_numpy = img_numpy.astype(np.uint8)
    if img_numpy.shape[2] == 3:
        img_opencv = cv.cvtColor(img_numpy, cv.COLOR_RGB2BGR)
    else:
        img_opencv = img_numpy

    return img_opencv


def read_image(path, input_type='numpy'):
    if path is None:
        raise ValueError('Empty path to the image')
    if not path.exists():
        raise ValueError('Incorrect path to the image')
    image = cv.imread(str(path))
    if image is None:
        raise ValueError(f'Failed to read image from {path}')

    if input_type == 'numpy':
        return image
    else:
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)
        return tensor.float() / 255.0


def save_image(img, save_path, input_type='numpy'):
    if img is None:
        raise ValueError('Empty image')
    if save_path is None:
        raise ValueError('Empty path to save')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if input_type == 'numpy':
        success = cv.imwrite(str(save_path), img)
    else:
        img_np = tensor_to_opencv(img)
        success = cv.imwrite(str(save_path), img_np)

    return success


def show_image(img, title='Result', input_type='numpy'):
    if img is None:
        raise ValueError('Empty image to show')

    if input_type == 'numpy':
        img_to_show = img
    else:
        img_to_show = tensor_to_opencv(img)
    img_height, img_width = img_to_show.shape[:2]

    monitors = get_monitors()
    win_width = min(img_width, monitors[0].width)
    win_height = min(img_height, monitors[0].height)

    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.resizeWindow(title, win_width, win_height)
    cv.imshow(title, img_to_show)
    cv.waitKey(0)
    cv.destroyAllWindows()
