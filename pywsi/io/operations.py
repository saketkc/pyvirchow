from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
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

import json
from matplotlib.patches import Polygon


def translate_and_scale_polygon(polygon,
                                x0,
                                y0,
                                scale_factor,
                                edgecolor='normal'):
    """Translate and then scales coordinates for polygon.

    Given the top-left coordinates of tile are at (x0, y0),
    and the polygon coordinates are being provided at level0,
    this method first translates the coordinates to origin by
    subtracting (x0, y0) from coordinates and then scales
    the coordinate to new level by using a scale factor.

    Example: If you are viewing an image at 1.25 magnification
    and the x0, y0 coordinates are (5,1000) at level0 magnification of 40x,
    then scale_factor = 1.25/40.

    Parameters
    ----------
    polygon: array_like
             Nx2 array of polygon coordinates

    x0, y0: int
            Top left start coordinates of patch

    scale_factor: float
                  Ratio of current level magnification to magnification at level0.

    Returns
    -------
    polygon: mpatches.Polygon
             object with scaled coordinates
    """
    if edgecolor == 'normal':
        edgecolor = '#00441b'
    elif edgecolor == 'tumor':
        edgecolor = '#ca0020'  #'#f03b20'

    polygon = polygon - np.array([x0, y0])
    polygon = polygon * scale_factor
    polygon = np.round(polygon).astype(int)
    polygon = Polygon(
        polygon, edgecolor=edgecolor, facecolor=None, fill=False, linewidth=4)
    return polygon


def draw_annotation(json_filepath, x0, y0, scale_factor, ax=None):
    """Draw manual annotations as in json file.

    Parameters
    ----------
    json_filepath: string
                   Path to json file containing polygon coordinates
    x0: int
        x coordinate of top left of patch
    y0: int
        y coordinate of top left of patch
    scale_factor: float
                  Scale coordinates by this (magnification/level0_magnification)
    ax: matploltib.axes
        If not None, add patches to this axis
    Returns
    -------
    polygon: array_like
             An array of mpathces.Polygon object containing apprpriately colored
             polygons

    Assumptions: x0, y0 are being provided at the level0 coordinates
    """
    json_parsed = json.load(open(json_filepath))
    tumor_patches = json_parsed['tumor']
    normal_patches = json_parsed['normal']
    polygons = []
    labelelled_polygons = []
    for index, tumor_patch in enumerate(tumor_patches):
        polygon = np.array(tumor_patch['vertices'])
        polygon = translate_and_scale_polygon(polygon, x0, y0, scale_factor,
                                              'tumor')
        if index == 0:
            polygon.set_label('tumor')
            labelelled_polygons.append(polygon)
        # For legend
        polygons.append(polygon)
    for index, normal_patch in enumerate(normal_patches):
        polygon = np.array(normal_patch['vertices'])
        polygon = translate_and_scale_polygon(polygon, x0, y0, scale_factor,
                                              'normal')
        if index == 0:
            polygon.set_label('normal')
            labelelled_polygons.append(polygon)
        polygons.append(polygon)
    if ax:
        for polygon in polygons:
            ax.add_patch(polygon)

        ax.legend(
            handles=labelelled_polygons,
            loc=9,
            bbox_to_anchor=(0.5, -0.1),
            ncol=2)
    return polygons


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
        patch_size: tuple or int
                    Patch size for renaming

        """
        if isinstance(patch_size, int):
            # If int provided, make a tuple
            patch_size = (patch_size, patch_size)
        if not patch_size:
            width, height = self.level_dimensions[level]
            # Adjust for the fact that the image top left
            # does not start at (0, 0), but is instead
            # moved to (xtart, ystart)
            width -= int(xstart * self.magnifications[level] / self.level0_mag)
            height -= int(
                ystart * self.magnifications[level] / self.level0_mag)
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
                  patch_size=None,
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
                                  patch_size=None,
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

    def get_annotation_bounding_boxes(self, json_filepath):
        """Get bounding boxes for manually annotated regions.

        Parameters
        ----------
        annotation_json: string
                         Path to annotation json

        Returns
        -------
        coordinates: array
                     coordinates in top-left -> top-right -> bottom-right -> bottom-left

        extreme_top_left_x, extreme_top_left_y: int, int
                                                Coordinates of the top left box.
                                                These coordinates can be used to
                                                get autofocused annotation patches.
                                                At least in principle.e
        """
        json_parsed = json.load(open(json_filepath))
        tumor_patches = json_parsed['tumor']
        normal_patches = json_parsed['normal']
        rectangles = OrderedDict()
        rectangles['tumor'] = []
        rectangles['normal'] = []
        extreme_top_left_x = self.dimensions[0]
        extreme_top_left_y = self.dimensions[1]
        for tumor_patch in tumor_patches:
            polygon = np.array(tumor_patch['vertices'])
            xmin, ymin = polygon.min(axis=0)
            xmax, ymax = polygon.max(axis=0)
            if xmin < extreme_top_left_x:
                extreme_top_left_x = xmin
            if ymin < extreme_top_left_y:
                extreme_top_left_y = ymin
            rectangle = OrderedDict()
            rectangle['top_left'] = (xmin, ymax)
            rectangle['top_right'] = (xmax, ymax)
            rectangle['bottom_right'] = (xmax, ymin)
            rectangle['bottom_left'] = (xmin, ymin)
            rectangles['tumor'].append(rectangle)
        for normal_patch in normal_patches:
            polygon = np.array(normal_patch['vertices'])
            xmin, ymin = polygon.min(axis=0)
            xmax, ymax = polygon.max(axis=0)
            rectangle = OrderedDict()
            rectangle['top_left'] = (xmin, ymax)
            rectangle['top_right'] = (xmax, ymax)
            rectangle['bottom_right'] = (xmax, ymin)
            rectangle['bottom_left'] = (xmin, ymin)
            rectangles['normal'].append(rectangle)
        return rectangles, (extreme_top_left_x, extreme_top_left_y)

    def autofocus_annotation(self,
                             json_filepath,
                             magnification=None,
                             level=None,
                             patch_size=None,
                             figsize=(10, 10)):
        """Autofocus on annotated patch by loading the extreme left
        of bounding box of the annotation

        """
        _, (xstart, ystart) = self.get_annotation_bounding_boxes(json_filepath)
        if not magnification and not level:
            raise ValueError(
                'Atleast one of magnification or level must be selected')
        return self.visualize_with_annotation(xstart, ystart, json_filepath,
                                              magnification, level, patch_size,
                                              figsize)
