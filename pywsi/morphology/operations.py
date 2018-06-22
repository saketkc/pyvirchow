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
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

from skimage.measure import label, regionprops

from skimage.segmentation import clear_border
import matplotlib.patches as mpatches
from skimage.color import label2rgb
from skimage.color import rgb2hsv

from ..io.operations import get_channel_hsv


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


def imopening(image, kernel_size=5, use_disk=True):
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


def imclosing(image, kernel_size=5, use_disk=True):
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
    opened = imopening(image, open_kernel_size, use_disk)
    closed = imclosing(opened, close_kernel_size, use_disk)
    return closed


def otsu_thresholding(rgb_image,
                      channel='saturation',
                      open_kernel_size=5,
                      close_kernel_size=5,
                      use_disk=True):
    """Perform OTSU thresholding followed by closing-then-opening

    rgb_image: np.uint8
               Input RGB image
    channel: string
             Channel on which to perform thresholding
    open_kernel_size: int
                      Size of opening kernel
    close_kernel_size: int
                       Size of closing kernel
    use_disk: bool
              Should use disk instead of a square
    """
    hsv_image = rgb2hsv(rgb_image)
    hsv_ch = get_channel_hsv(hsv_image, channel)
    otsu = threshold_otsu(hsv_ch)
    thresholded = hsv_ch > otsu

    close_then_open = open_close(thresholded, open_kernel_size,
                                 close_kernel_size, use_disk)
    return close_then_open


def plot_contours(bw_image, rgb_image, ax=None):
    """Plot contours over a otsu thresholded binary image.

    Parameters
    ----------
    bw_image: np.uint8
              Input
    """
    cleared = clear_border(bw_image)
    label_image = label(cleared)
    image_label_overlay = label2rgb(label_image, image=rgb_image)

    if not ax:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
    ax.imshow(image_label_overlay)
    for region in regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle(
            (minc, minr),
            maxc - minc,
            fill=False,
            edgecolor='red',
            linewidth=2)
        ax.add_patch(rect)
    ax.set_axis_off()
    fig.tight_layout()
    return ax
