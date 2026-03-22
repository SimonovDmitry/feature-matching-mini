from pathlib import Path
import cv2 as cv
import numpy as np
from numpy import ndarray
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

def read_image(path: str, input_type: str = 'numpy') -> ndarray | torch.Tensor:
    if path is None:
        raise ValueError('Empty path to the image')
    filepath = Path(path)
    if not filepath.exists():
        raise ValueError('Incorrect path to the image')
    image = cv.imread(path)
    if image is None:
        raise ValueError(f'Failed to read image from {path}')

    if input_type == 'numpy':
        return image
    else:
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)
        return tensor.float() / 255.0

def save_image(img: ndarray | torch.Tensor, save_path: str, input_type: str = 'numpy') -> bool:
    if img is None:
        raise ValueError('Empty image')
    if save_path is None:
        raise ValueError('Empty path to save')
    filepath = Path(save_path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if input_type == 'numpy':
        success = cv.imwrite(save_path, img)
    else:
        img_np = tensor_to_opencv(img)
        success = cv.imwrite(save_path, img_np)

    return success

def show_image(img: ndarray | torch.Tensor, title: str = 'Result',
               input_type: str = 'numpy') -> None:
    if img is None:
        raise ValueError('Empty image to show')

    if input_type == 'numpy':
        img_to_show = img
    else:
        img_to_show = tensor_to_opencv(img)
    img_height, img_width = img_to_show.shape[:2]

    cv.namedWindow('temp', cv.WINDOW_NORMAL)
    cv.setWindowProperty('temp', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    screen_width = cv.getWindowImageRect('temp')[2]
    screen_height = cv.getWindowImageRect('temp')[3]
    cv.destroyWindow('temp')

    win_width = min(img_width, screen_width)
    win_height = min(img_height, screen_height)

    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.resizeWindow(title, win_width, win_height)
    cv.imshow(title, img_to_show)
    cv.waitKey(0)
    cv.destroyAllWindows()