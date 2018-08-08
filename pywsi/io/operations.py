from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
import os
import six
import ntpath
import warnings
import openslide
from openslide import OpenSlide
import matplotlib
matplotlib.use('Agg', warn=False)
import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.color import rgb2hsv
from skimage.color import rgb2lab

import json
from shapely.geometry import Polygon as shapelyPolygon
from matplotlib.patches import Polygon, Rectangle
from PIL import Image, ImageDraw


def rectangle_dict_to_mpl(rectangle_dict, x0, y0, scale_factor, edgecolor):
    """Convert rectangle dict to matplotlib patch Rectangle

    Parameters
    ----------
    rectangle_dict: dict
                    dict with keys as defined in get_annotation_bounding_boxes
    """
    if edgecolor == 'normal':
        edgecolor = '#00441b'
    elif edgecolor == 'tumor':
        edgecolor = '#ca0020'  #'#f03b20'
    xmin, ymin = rectangle_dict['top_left']
    xmax, ymax = rectangle_dict['bottom_right']

    xleft = int((xmin - x0) * scale_factor)
    yleft = int((ymin - y0) * scale_factor)

    width = int((xmax - xmin) * scale_factor)
    height = int((ymax - ymin) * scale_factor)
    assert width > 0, 'width should be > 0'
    assert height > 0, 'height should be > 0'
    return Rectangle(
        (xleft, yleft),
        width,
        height,
        edgecolor=edgecolor,
        facecolor=None,
        fill=False,
        linewidth=4)


def poly2mask(polygons, shape):
    """Create mask from coordinates.

    Parameters
    ----------
    polygons: array_like
              Array of vertices [(x1,y1), (x2, y2)]
    shape: tuple
           (width, height)

    Returns
    -------
    mask: np.uint8
          Boolean masked array
    """

    img = Image.new('L', shape, 0)
    draw = ImageDraw.Draw(img)
    for polygon in polygons:
        if isinstance(polygon, shapelyPolygon):
            coordinates = list(polygon.boundary.coords)
        else:
            coordinates = polygon.get_xy()
        coordinates_int = []
        for x, y in coordinates:
            coordinates_int.append((int(x), int(y)))
        draw.polygon(coordinates_int, outline=1, fill=1)
    mask = np.array(img)
    return mask


def translate_and_scale_object(array, x0, y0, scale_factor):
    """Translate and scale any object

    Parameters
    ----------
    arrayn: array_like
            Nx2 array of polygon coordinates

    x0, y0: int
            Top left start coordinates of patch

    scale_factor: float
                  Ratio of current level magnification to magnification at level0.

    """
    if isinstance(array, Polygon):
        array = np.array(array.get_xy())
    elif isinstance(array, shapelyPolygon):
        array = np.array(array.exterior.coords)
    array = array - np.array([x0, y0])
    array = array * scale_factor
    array = np.round(array).astype(int)
    return array


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
        edgecolor = '#2ca25f'  # '#00441b'
    elif edgecolor == 'tumor':
        edgecolor = '#ca0020'  #'#f03b20'
    polygon = translate_and_scale_object(polygon, x0, y0, scale_factor)
    polygon = Polygon(
        polygon, edgecolor=edgecolor, facecolor=None, fill=False, linewidth=2)
    return polygon


def get_annotation_bounding_boxes(json_filepath):
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
    for tumor_patch in tumor_patches:
        polygon = np.array(tumor_patch['vertices'])
        xmin, ymin = polygon.min(axis=0)
        xmax, ymax = polygon.max(axis=0)
        rectangle = OrderedDict()
        rectangle['top_left'] = (xmin, ymin)
        rectangle['top_right'] = (xmax, ymin)
        rectangle['bottom_right'] = (xmax, ymax)
        rectangle['bottom_left'] = (xmin, ymax)
        rectangles['tumor'].append(rectangle)
    for normal_patch in normal_patches:
        polygon = np.array(normal_patch['vertices'])
        xmin, ymin = polygon.min(axis=0)
        xmax, ymax = polygon.max(axis=0)
        rectangle = OrderedDict()
        rectangle['top_left'] = (xmin, ymin)
        rectangle['top_right'] = (xmax, ymin)
        rectangle['bottom_right'] = (xmax, ymax)
        rectangle['bottom_left'] = (xmin, ymax)
        rectangles['normal'].append(rectangle)
    return rectangles


def get_annotation_polygons(json_filepath, polygon_type='mpl'):
    """Get annotation jsons polygons

    Assumed to be at level 0

    Parameters
    ----------
    json_filepath: string
                    Path to json file
    polygon_type: string
                  'matplotlib/shapely'

    Returns
    -------
    polygons: dict(Polygons)
                dict of matplotlib.Polygons with keys are normal/tumor

    """
    json_parsed = json.load(open(json_filepath))
    tumor_patches = json_parsed['tumor']
    normal_patches = json_parsed['normal']
    polygons = OrderedDict()
    polygons['tumor'] = []
    polygons['normal'] = []
    for tumor_patch in tumor_patches:
        tumor_patch['vertices'] = np.array(tumor_patch['vertices'])
        if polygon_type == 'mpl':
            polygon = Polygon(tumor_patch['vertices'])
        else:
            if tumor_patch['vertices'].shape[0] >= 3:
                polygon = shapelyPolygon(tumor_patch['vertices'])
            else:
                continue
        polygons['tumor'].append(polygon)
    for normal_patch in normal_patches:
        if polygon_type == 'mpl':
            polygon = Polygon(np.array(tumor_patch['vertices']))
        else:
            polygon = shapelyPolygon(np.array(tumor_patch['vertices']))
        polygons['normal'].append(polygon)
    return polygons


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


def draw_annotation_boxes(json_filepath, x0, y0, scale_factor, ax=None):
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
    rectangles: array_lik
                An array of mpathces.Polygon object containing apprpriately colored rectangles

    Assumptions: x0, y0 are being provided at the level0 coordinates
    """
    bounding_boxes = get_annotation_bounding_boxes(json_filepath)
    tumor_boxes = bounding_boxes['tumor']
    normal_boxes = bounding_boxes['normal']
    rectangles = []
    labelelled_rectangles = []
    for index, tumor_box in enumerate(tumor_boxes):
        rectangle = rectangle_dict_to_mpl(tumor_box, x0, y0, scale_factor,
                                          'tumor')
        if index == 0:
            rectangle.set_label('tumor')
            labelelled_rectangles.append(rectangle)
        # For legend
        rectangles.append(rectangle)
    for index, normal_box in enumerate(normal_boxes):
        rectangle = rectangle_dict_to_mpl(normal_box, x0, y0, scale_factor,
                                          'normal')
        if index == 0:
            rectangle.set_label('normal')
            labelelled_rectangles.append(rectangle)
        # For legend
        rectangles.append(rectangle)
    if ax:
        for rectangle in rectangles:
            ax.add_patch(rectangle)

        ax.legend(
            handles=labelelled_rectangles,
            loc=9,
            bbox_to_anchor=(0.5, -0.1),
            ncol=2)
    return rectangles


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

        self.uid = os.path.splitext(path_leaf(image_path))[0]
        self.filepath = image_path
        width, height = self.dimensions
        self.width = width
        self.height = height
        self.magnifications = [
            self.level0_mag / downsample
            for downsample in self.level_downsamples
        ]

    def get_patch_by_level(self, x0, y0, level, patch_size=None):
        """Get patch by specifying magnification

        Parameters
        ----------
        x0: int
                Top left pixel x coordinate
        y0: int
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
            # moved to (xtart, y0)
            width -= int(x0 * self.magnifications[level] / self.level0_mag)
            height -= int(y0 * self.magnifications[level] / self.level0_mag)
        else:
            width, height = patch_size
        patch = self.read_region((x0, y0), level,
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
        return scale_factor, possible_mag

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
                                   x0,
                                   y0,
                                   magnification,
                                   patch_size=None):
        """Get patch by specifying magnification

        Parameters
        ----------
        x0: int
                Top left pixel x coordinate
        y0: int
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
            (x0, y0), possible_level,
            (rescaled_width, rescaled_height)).convert('RGB')
        return np.array(patch)

    def show_all_properties(self):
        """Print all properties.
        """
        print('Properties:')
        for key in self.properties.keys():
            print('{} : {}'.format(key, self.properties[key]))

    def visualize(self,
                  x0,
                  y0,
                  magnification=None,
                  level=None,
                  patch_size=None,
                  figsize=(10, 10)):
        """Visualize patch.

        x0: int
                X coordinate of top left corner of patch
        y0: int
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
            patch = self.get_patch_by_magnification(x0, y0, magnification,
                                                    patch_size)
        else:
            patch = self.get_patch_by_level(x0, y0, level, patch_size)
        return imshow(patch, figsize=figsize)

    def visualize_with_annotation(self,
                                  x0,
                                  y0,
                                  annotation_json,
                                  magnification=None,
                                  level=None,
                                  patch_size=None,
                                  show_boundary=True,
                                  show_box=False,
                                  figsize=(10, 10)):
        """Visualize patch with manually annotated regions marked in red/green.

        Parameters
        ----------
        x0, y0: int
                Coordinates of top-left of patch
        annotation_json: string
                         Path to json containing annotated coordinates
        magnification: float
                       Magnification
        level: int
               0-9 level with 0 being the highest zoom level
        patch_size: int
                    Patch size to extract (the final size might not necessarily be this)
        show_boundary: bool
                       Should draw the annotation boundaries
        show_box: bool
                  Should draw a box around the annotation
        """
        if not magnification and not level:
            raise ValueError(
                'Atleast one of magnification or level must be selected')
        if magnification:
            patch = self.get_patch_by_magnification(x0, y0, magnification,
                                                    patch_size)
            scale_factor = self.get_mag_scale_factor(magnification)
        else:
            patch = self.get_patch_by_level(x0, y0, level, patch_size)
            scale_factor = self.get_level_scale_factor(level)

        ax = imshow(patch, figsize=figsize)
        if show_boundary:
            annotation_polygons = draw_annotation(annotation_json, x0, y0,
                                                  scale_factor, ax)
            self.annotation_polygons = annotation_polygons
        if show_box:
            draw_annotation_boxes(annotation_json, x0, y0, scale_factor, ax)

    def autofocus_annotation(self,
                             json_filepath,
                             magnification=None,
                             level=None,
                             patch_size=None,
                             figsize=(10, 10)):
        """Autofocus on annotated patch by loading the extreme left
        of bounding box of the annotation

        """
        bounding_boxes = get_annotation_bounding_boxes(json_filepath)
        extreme_top_left_x = self.dimensions[0]
        extreme_top_left_y = self.dimensions[1]
        for xbox in bounding_boxes.values():
            for box in xbox:
                xmin, ymin = box['top_left']
                xmax, ymax = box['bottom_right']
                if xmin < extreme_top_left_x:
                    extreme_top_left_x = xmin
                if ymin < extreme_top_left_y:
                    extreme_top_left_y = ymin

        x0, y0 = extreme_top_left_x, extreme_top_left_y
        if not magnification and not level:
            raise ValueError(
                'Atleast one of magnification or level must be selected')
        return self.visualize_with_annotation(
            x0, y0, json_filepath, magnification, level, patch_size, figsize)

    def annotation_masked(self,
                          json_filepath,
                          magnification=None,
                          level=None,
                          savedir=None):
        """Create a masked image for the annotated regions.

        """
        x0 = 0
        y0 = 0
        patch_size = None
        if magnification is None and level is None:
            raise ValueError(
                'Atleast one of magnification or level must be selected')
        if magnification:
            patch = self.get_patch_by_magnification(x0, y0, magnification,
                                                    patch_size)
            scale_factor, possible_mag = self.get_mag_scale_factor(
                magnification)
            level = self.magnifications.index(possible_mag)
        else:
            patch = self.get_patch_by_level(x0, y0, level, patch_size)
            scale_factor = self.get_level_scale_factor(level)
        shape = patch.shape
        shape = self.level_dimensions[level]
        json_parsed = json.load(open(json_filepath))
        tumor_patches = json_parsed['tumor']
        normal_patches = json_parsed['normal']
        normal_polygons = []
        tumor_polygons = []
        for index, tumor_patch in enumerate(tumor_patches):
            polygon = np.array(tumor_patch['vertices'])
            polygon = translate_and_scale_polygon(polygon, x0, y0,
                                                  scale_factor, 'tumor')
            tumor_polygons.append(polygon)
        for index, normal_patch in enumerate(normal_patches):
            polygon = np.array(normal_patch['vertices'])
            polygon = translate_and_scale_polygon(polygon, x0, y0,
                                                  scale_factor, 'normal')
            normal_polygons.append(polygon)
        tumor_mask = poly2mask(tumor_polygons, (shape[0], shape[1]))
        normal_mask = poly2mask(normal_polygons, (shape[0], shape[1]))
        combined_mask = poly2mask(normal_polygons + tumor_polygons,
                                  (shape[0], shape[1]))
        if savedir:
            ID = self.uid.replace('.tif', '')
            os.makedirs(savedir, exist_ok=True)
            tumor_filepath = os.path.join(savedir,
                                          ID + '_AnnotationTumorMask.npy')
            np.save(tumor_filepath, tumor_mask)
            normal_filepath = os.path.join(savedir,
                                           ID + '_AnnotationNormalMask.npy')
            np.save(normal_filepath, normal_mask)
            combined_filepath = os.path.join(
                savedir, ID + '_AnnotationCombinedMask.npy')
            np.save(combined_filepath, combined_mask)
            colored_filepath = os.path.join(savedir,
                                            ID + '_AnnotationColored.npy')
            np.save(colored_filepath, patch)

        return tumor_mask, normal_mask, combined_mask

    def __getstate__(self):

        return [
            self.level0_mag, self.uid, self.filepath, self.dimensions,
            self.width, self.height, self.magnifications
        ]
