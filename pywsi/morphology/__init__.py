from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from skimage.morphology import closing
from skimage.morphology import dilation
from skimage.morphology import erosion
from skimage.morphology import opening
from skimage.morphology import disk

import numpy as np


def _get_kernel(kernel_size, use_disk=True):
    if not use_disk:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
    else:
        kernel = disk(kernel_size)
    return kernel


def imerode(image, kernel_size=5, use_disk=True):
    """Erode an image.

    Parameters
    ----------
    image: array_like
           np.uint8 binary thresholded image
    kernel_size: int
                 Integer
    use_disk: bool
              Use disk instead of a NxN kernel

    Returns
    -------
    eroded: array_like
            np.uint8 eroded image
    """
    kernel = _get_kernel(kernel_size, use_disk)
    eroded = erosion(image, kernel)
    return eroded


def imdilate(image, kernel_size=5, use_disk=True):
    """Dilate an image

    Parameters
    ----------
    image: array_like
           np.uint8 binary thresholded image
    kernel_size: int
                 Integer

    Returns
    -------
    dilated: array_like
             np.uint8 eroded image
    """
    kernel = _get_kernel(kernel_size, use_disk)
    dilated = dilation(image, kernel)
    return dilated


def imopen(image, kernel_size=5, use_disk=True):
    """Open an image.

    Parameters
    ----------
    image: array_like
           np.uint8 binary thresholded image
    kernel_size: int
                 Integer

    Returns
    -------
    opened: array_like
            np.uint8 opened
    """
    kernel = _get_kernel(kernel_size, use_disk)
    opened = opening(image, kernel)
    return opened


def imclose(image, kernel_size=5, use_disk=True):
    """Close an image.

    Parameters
    ----------
    image: array_like
           np.uint8 binary thresholded image
    kernel_size: int
                 Integer

    Returns
    -------
    closed: array_like
            np.uint8 opened
    """
    kernel = _get_kernel(kernel_size, use_disk)
    closed = closing(image, kernel)
    return closed


def open_close(image, open_kernel_size=5, close_kernel_size=5, use_disk=True):
    """Open followed by closing an image.

    Parameters
    ----------
    image: array_like
           np.uint8 binary thresholded image
    open_kernel_size: int
                      Integer
    close_kernel_size: int
                       Integer

    Returns
    -------
    closed: array_like
            np.uint8 opened-closed
    """
    opened = imopen(image, open_kernel_size, use_disk)
    closed = imclose(opened, close_kernel_size, use_disk)
    return closed
