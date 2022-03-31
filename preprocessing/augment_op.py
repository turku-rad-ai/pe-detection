import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

"""
This module contains image augmentation operations that can be added to
config.yml by using functions' names. These operations are fairly conservative
in order to keep the data as realistic as possible.
"""

SEED = 1


def blur(im: np.ndarray, sigma: int = 3) -> np.ndarray:
    im_blur = cv2.GaussianBlur(im, (sigma, sigma), 0)

    return im_blur


def zoom(im: np.ndarray, zoom_factor: float = 1.15) -> np.ndarray:
    zoom_factor = np.fmax(1.0, zoom_factor)
    height, width = im.shape[:2]

    center_x, center_y = width // 2, height // 2
    radius_x, radius_y = int((1.0 / zoom_factor) * width / 2), int((1.0 / zoom_factor) * height / 2)

    min_x, max_x = center_x - radius_x, center_x + radius_x
    min_y, max_y = center_y - radius_y, center_y + radius_y

    im_cropped = im[min_x:max_x, min_y:max_y]
    im_resized_cropped = cv2.resize(im_cropped, (width, height))

    return im_resized_cropped


def zoom_1_05(im: np.ndarray) -> np.ndarray:
    zoom_factor = 1.05
    return zoom(im, zoom_factor)


def zoom_1_075(im: np.ndarray) -> np.ndarray:
    zoom_factor = 1.075
    return zoom(im, zoom_factor)


def zoom_1_15(im: np.ndarray) -> np.ndarray:
    zoom_factor = 1.15
    return zoom(im, zoom_factor)


def translate(im: np.ndarray, tx: float = 0, ty: float = 0) -> np.ndarray:
    M = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
    im_trans = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]))

    return im_trans


def tr_x10(im: np.ndarray) -> np.ndarray:
    return translate(im, tx=10, ty=0)


def tr_x15(im: np.ndarray) -> np.ndarray:
    return translate(im, tx=15, ty=0)


def tr_x20(im: np.ndarray) -> np.ndarray:
    return translate(im, tx=20, ty=0)


def tr_xm10(im: np.ndarray) -> np.ndarray:
    return translate(im, tx=-10, ty=0)


def tr_xm15(im: np.ndarray) -> np.ndarray:
    return translate(im, tx=-15, ty=0)


def tr_xm20(im: np.ndarray) -> np.ndarray:
    return translate(im, tx=-20, ty=0)


def tr_y10(im: np.ndarray) -> np.ndarray:
    return translate(im, tx=0, ty=10)


def tr_y15(im: np.ndarray) -> np.ndarray:
    return translate(im, tx=0, ty=15)


def tr_y20(im: np.ndarray) -> np.ndarray:
    return translate(im, tx=0, ty=20)


def tr_ym10(im: np.ndarray) -> np.ndarray:
    return translate(im, tx=0, ty=-10)


def tr_ym15(im: np.ndarray) -> np.ndarray:
    return translate(im, tx=0, ty=-15)


def tr_ym20(im: np.ndarray) -> np.ndarray:
    return translate(im, tx=0, ty=-20)


def rotate(im: np.ndarray, angle: int = 10) -> np.ndarray:
    M = cv2.getRotationMatrix2D((im.shape[1] / 2, im.shape[0] / 2), angle, 1)
    im_rot = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]))

    return im_rot


def rotate_3(im: np.ndarray) -> np.ndarray:
    return rotate(im, angle=3)


def rotate_m3(im: np.ndarray) -> np.ndarray:
    return rotate(im, angle=-3)


def rotate_5(im: np.ndarray) -> np.ndarray:
    return rotate(im, angle=5)


def rotate_m5(im: np.ndarray) -> np.ndarray:
    return rotate(im, angle=-5)


def gaussian_noise(
    im: np.ndarray, mean: float = 0, std: float = 5, grayscale: bool = True
) -> np.ndarray:
    """Add gaussian noise to unsigned integer valued image"""
    random_state = np.random.RandomState(SEED)

    if grayscale:
        gaus_noise = random_state.normal(mean, std, (im.shape[0], im.shape[1], 1))
    else:
        gaus_noise = random_state.normal(mean, std, (im.shape))

    flt_image = im.astype(np.float32)

    noisy_image = np.maximum(np.zeros(im.shape), flt_image + gaus_noise)
    noisy_image = np.minimum(np.full(im.shape, np.iinfo(im.dtype).max), noisy_image)
    noisy_image = noisy_image.astype(im.dtype)

    return noisy_image


def elastic_transform(im: np.ndarray, alpha: float = 1000, sigma: float = 10) -> np.ndarray:
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.

       Adapted from https://gist.github.com/fmder/e28813c1e8721830ff9c
    """
    random_state = np.random.RandomState(SEED)

    shape = im.shape
    rand_matrix_x = random_state.rand(*shape) * 2 - 1
    rand_matrix_y = random_state.rand(*shape) * 2 - 1

    dx = gaussian_filter(rand_matrix_x, sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(rand_matrix_y, sigma, mode="constant", cval=0) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(im, indices, order=1, mode="reflect")

    return distored_image.reshape(shape)
