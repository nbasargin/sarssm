"""
Common functions, helpers, and utilities.
"""

import numpy as np
from scipy.ndimage import uniform_filter


def _multilook(img, window_size):
    return uniform_filter(img, window_size, mode="constant", cval=0)

def _abs_squared(img):
    return img.real**2 + img.imag**2  # equivalent to np.abs(img) ** 2

def complex_coherence(img_1, img_2, window_size):
    """
    Estimate the complex coherence of two complex images.
    Window size can either be a single number or a tuple with two numbers.
    """
    interferogram = _multilook(img_1 * np.conj(img_2), window_size)
    abs_sqr_img_1 = _multilook(_abs_squared(img_1), window_size)
    abs_sqr_img_2 = _multilook(_abs_squared(img_2), window_size)
    return interferogram / np.sqrt(abs_sqr_img_1 * abs_sqr_img_2)
