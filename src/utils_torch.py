import cv2 as cv
import numpy as np
import torch

def is_tensor(data):
    return isinstance(data, torch.Tensor)


def empty(shape, dtype=torch.float32, device='cpu'):
    return torch.empty(shape, dtype=dtype, device=device)


def get_device(device=None):
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device('cuda')

    if torch.backends.mps.is_available():
        return torch.device('mps')

    return torch.device('cpu')


def numpy_to_tensor(data, device='cpu', dtype=None):
    array = np.ascontiguousarray(data)
    tensor = torch.from_numpy(array)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device)


def tensor_to_numpy(data, dtype=None):
    if not is_tensor(data):
        return data

    array = data.detach().cpu().numpy()
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def image_to_tensor(data, device='cpu'):
    if data.ndim == 2:
        img_rgb = cv.cvtColor(data, cv.COLOR_GRAY2RGB)
    elif data.ndim == 3 and data.shape[2] == 3:
        img_rgb = cv.cvtColor(data, cv.COLOR_BGR2RGB)
    else:
        img_rgb = data

    tensor = torch.from_numpy(np.ascontiguousarray(img_rgb)).to(torch.float32)

    if tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(2, 0, 1)

    return (tensor / 255.0).to(device)


def features_to_tensor(data, device='cpu'):
    keypoints = cv.KeyPoint_convert(data.get('kp'))
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.size == 0:
        keypoints = np.empty((0, 2), dtype=np.float32)

    descriptors = data.get('des')
    if descriptors is None:
        descriptors = np.empty((len(keypoints), 0), dtype=np.float32)
    else:
        descriptors = np.asarray(descriptors)

    result = {
        'keypoints': numpy_to_tensor(keypoints, device=device),
        'descriptors': numpy_to_tensor(descriptors, device=device),
    }

    if 'width' in data:
        result['width'] = data['width']
    if 'height' in data:
        result['height'] = data['height']

    return result


def matches_to_tensor(data, device='cpu'):
    dmatches = data.get('matches')

    if not dmatches:
        matches_np = np.empty((0, 2), dtype=np.int64)
    else:
        matches_np = np.asarray(
            [[match.queryIdx, match.trainIdx] for match in dmatches],
            dtype=np.int64,
        ).reshape(-1, 2)

    return {
        'matches': numpy_to_tensor(
            matches_np,
            device=device,
            dtype=torch.long,
        )
    }
