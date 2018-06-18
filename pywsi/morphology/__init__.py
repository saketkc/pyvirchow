import cv2
import numpy as np


def imerode(image, kernel_size=5):
    """Erode an image.

    Parameters
    ----------
    image: array_like
           np.uint8 binary thresholded image
    kernel_size: int
                 Integer

    Returns
    -------
    eroded: array_like
            np.uint8 eroded image
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=1)
    return eroded


def imdilate(image, kernel_size=5):
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
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    return dilated


def imopen(image, kernel_size):
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
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened


def imclose(image, kernel_size):
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
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return closed


def open_close(image, open_kernel_size=5, close_kernel_size=5):
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
    opened = imopen(image, open_kernel_size)
    closed = imclose(opened, close_kernel_size)
    return closed
