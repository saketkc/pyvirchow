from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import ntpath
import warnings
import openslide
from openslide import OpenSlide
import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.color import rgb2hsv
from skimage.color import rgb2lab

from ..morphology.mask import draw_annotation


def path_leaf(path):
    """Get base and tail of filepath

    Parameters
    ----------
    path: string
          Path to split

    Returns
    -------
    tail: string
          Get tail of path
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def read_as_rgb(image_path):
    """Read image as RGB

    Parameters
    ----------
    image_path: str
                Path to image

    Returns
    -------
    rgb_image: array_like
               np.uint8 array of RGB values
    """
    image = imread(image_path)
    return image


def read_as_lab(image):
    """Read image as RGB

    Parameters
    ----------
    imag: str
                Path to image

    Returns
    -------
    lab_image: array_like
               np.uint8 array of LAB values
    """
    if isinstance(image, six.string_types):
        image = read_as_rgb(image)
    image = rgb2lab(image)
    return image


def read_as_hsv(image):
    """Convert RGB image to HSV

    Parameters
    ----------
    image: array_like or str
           np.uint8 array of RGB values or image path

    Returns
    -------
    hsv_image: array_like
               np.uint8 array of HSV values
    """
    if isinstance(image, six.string_types):
        image = read_as_rgb(image)
    hsv_image = rgb2hsv(image)
    return hsv_image


def read_as_gray(image):
    """Convert RGB image to gray

    Parameters
    ----------
    image: array_like or str
           np.uint8 array of RGB values or image path

    Returns
    -------
    gray_image: array_like
               np.uint8 array of HSV values
    """
    if isinstance(image, six.string_types):
        image = read_as_rgb(image)
    gray_image = rgb2gray(image)
    return gray_image


def imshow(image, is_gray=True, ax=None, figsize=(10, 10)):
    """Visualize an rgb image.

    Parameters
    ----------
    rgb_image: array_like
               np.unit8 array
    ax: matplotlib.Axes
        Axis object
    figsize: tuple
             (width, height) of figure

    Returns
    -------
    imshow: plt.imshow
            Can only be visualized in notebook
    """
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    if is_gray:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)
    ax.set_axis_off()
    fig.tight_layout()
    return ax


class WSIReader(OpenSlide):
    def __init__(self, image_path, level0_mag):
        """Read image as Openslide object

        Parameters
        ----------
        image_path: str
                    Path to image file

        level0_mag: int
                    Magnification at level 0 (most detailed)
        """
        super(WSIReader, self).__init__(image_path)
        inferred_level0_mag = self.properties.get(
            openslide.PROPERTY_NAME_OBJECTIVE_POWER, None)
        if inferred_level0_mag:
            if inferred_level0_mag != level0_mag:
                warnings.warn(
                    '''Inferred and provided level0 mag mismatch.
                              Provided {}, but found {}.
                    Will use the latter.'''.format(inferred_level0_mag,
                                                   level0_mag), UserWarning)
                self.level0_mag = level0_mag
        else:
            self.level0_mag = level0_mag

        self.uid = path_leaf(image_path)
        self.filepath = image_path
        width, height = self.dimensions
        self.width = width
        self.height = height
        self.magnifications = [
            self.level0_mag / downsample
            for downsample in self.level_downsamples
        ]

    def get_patch_by_level(self, xstart, ystart, level, patch_size=None):
        """Get patch by specifying magnification

        Parameters
        ----------
        xstart: int
                Top left pixel x coordinate
        ystart: int
                Top left pixel y coordinate
        magnification: int
                       Magnification to extract at
        patch_size: tuple
                    Patch size for renaming

        """
        if not patch_size:
            width, height = self.level_dimensions[level]
            print('patch  width, height: {} {}'.format(width, height))
        else:
            width, height = patch_size
        patch = self.read_region((xstart, ystart), level,
                                 (width, height)).convert('RGB')
        return np.array(patch)

    def get_mag_scale_factor(self, magnification):
        """Given a magnification, get scale factor.

        If the magnification is not in the list of possible magnifications,
        get the next highest possible magnification.

        Parameters
        ----------
        magnification: float
                       Desired magnification

        Returns
        -------
        scale_factor: float
                      Corresponding scale factor
        """
        filtered_mag = list(
            filter(lambda x: x >= magnification, self.magnifications))
        # What is the possible magnification available?
        possible_mag = min(filtered_mag)
        scale_factor = possible_mag / self.level0_mag
        return scale_factor

    def get_level_scale_factor(self, level):
        """Given a level, get scale factor.

        Parameters
        ----------
        level: float
               Desired level

        Returns
        -------
        scale_factor: float
                      Corresponding scale factor
        """
        return self.magnifications[level] / self.level0_mag

    def get_patch_by_magnification(self,
                                   xstart,
                                   ystart,
                                   magnification,
                                   patch_size=None):
        """Get patch by specifying magnification

        Parameters
        ----------
        xstart: int
                Top left pixel x coordinate
        ystart: int
                Top left pixel y coordinate
        magnification: int
                       Magnification to extract at
        patch_size: tuple
                    Patch size for renaming

        """
        filtered_mag = list(
            filter(lambda x: x >= magnification, self.magnifications))
        # What is the possible magnification available?
        possible_mag = min(filtered_mag)
        possible_level = self.magnifications.index(possible_mag)
        # Rescale the size of image to match the new magnification
        if patch_size:
            rescaled_size = possible_mag / magnification * patch_size
            rescaled_width, rescaled_height = int(rescaled_size), int(
                rescaled_size)
        else:
            rescaled_width, rescaled_height = self.level_dimensions[
                possible_level]
        patch = self.read_region(
            (xstart, ystart), possible_level,
            (rescaled_width, rescaled_height)).convert('RGB')
        return np.array(patch)

    def show_all_properties(self):
        """Print all properties.
        """
        print('Properties:')
        for key in self.properties.keys():
            print('{} : {}'.format(key, self.properties[key]))

    def visualize(self,
                  xstart,
                  ystart,
                  magnification=None,
                  level=None,
                  patch_size=1000,
                  figsize=(10, 10)):
        """Visualize patch.

        xstart: int
                X coordinate of top left corner of patch
        ystart: int
                Y coordinate of top left corner of patch
        magnification: int
                       If provided, uses a magnification
        level: int
               What level of pyramid to use
        patch_size: int
                    Size of Patch
        """
        if not magnification and not level:
            raise ValueError(
                'Atleast one of magnification or level must be selected')
        if magnification:
            patch = self.get_patch_by_magnification(xstart, ystart,
                                                    magnification, patch_size)
        else:
            patch = self.get_patch_by_level(xstart, ystart, level, patch_size)
        return imshow(patch, figsize=figsize)

    def visualize_with_annotation(self,
                                  xstart,
                                  ystart,
                                  annotation_json,
                                  magnification=None,
                                  level=None,
                                  patch_size=1000,
                                  figsize=(10, 10)):
        if not magnification and not level:
            raise ValueError(
                'Atleast one of magnification or level must be selected')
        if magnification:
            patch = self.get_patch_by_magnification(xstart, ystart,
                                                    magnification, patch_size)
            scale_factor = self.get_mag_scale_factor(magnification)
        else:
            patch = self.get_patch_by_level(xstart, ystart, level, patch_size)
            scale_factor = self.get_level_scale_factor(level)

        ax = imshow(patch, figsize=figsize)
        annotation_polygons = draw_annotation(annotation_json, xstart, ystart,
                                              scale_factor, ax)
        self.annotation_polygons = annotation_polygons
