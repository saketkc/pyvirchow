from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import ntpath
import warnings
import cv2
import openslide
from openslide import OpenSlide
import matplotlib.pyplot as plt


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
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_as_lab(image_path):
    """Read image as RGB

    Parameters
    ----------
    image_path: str
                Path to image

    Returns
    -------
    lab_image: array_like
               np.uint8 array of LAB values
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return image


def rgb_to_hsv(image):
    """Convert RGB image to HSV

    Parameters
    ----------
    image: array_like
           np.uint8 array of RGB values

    Returns
    -------
    hsv_image: array_like
               np.uint8 array of HSV values
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return hsv_image


def rgb_to_lab(image):
    """Convert RGB image to LAB coordinates

    Parameters
    ----------
    image: array_like
           np.uint8 array of RGB values

    Returns
    -------
    lab_image: array_like
               np.float32 array of LAB values
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab_image


def lab_to_rgb(image):
    """Convert LAB image to RGB coordinates

    Parameters
    ----------
    image: array_like
           np.float32 array of LAB values

    Returns
    -------
    rgb_image: array_like
               np.uint8 array of LAB values
    """
    rgb_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return rgb_image


def imshow(image, ax=None, figsize=(10, 10)):
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
    ax.imshow(image)
    ax.axis('off')
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
                              Provided {}, but found {}. Will use the latter.'''
                    .format(inferred_level0_mag, level0_mag), UserWarning)
                self.level0_mag = level0_mag
        else:
            self.level0_mag = level0_mag

        self.uid = path_leaf(image_path)
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
        else:
            width, height = patch_size
        patch = self.read_region((xstart, ystart), level,
                                 (width, height)).convert('RGB')
        return patch

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
        return patch

    def show_all_properties(self):
        """Print all properties.
        """
        print('Properties')
        for key in self.properties.keys():
            print('{} : {}'.format(key, self.properties[key]))

    def visualize(self,
                  xstart,
                  ystart,
                  magnification=None,
                  level=None,
                  patch_size=1000):
        if not magnification and not level:
            raise ValueError(
                'Atleast one of magnification or level must be selected')
        if magnification:
            patch = self.get_patch_by_magnification(xstart, ystart,
                                                    magnification, patch_size)
        else:
            patch = self.get_patch_by_level(xstart, ystart, level, patch_size)
        return imshow(patch)
