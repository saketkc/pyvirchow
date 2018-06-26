from skimage import draw
import numpy as np
import json
from matplotlib.patches import Polygon


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    """Create mask from coordinates.

    Parameters
    ----------
    vertex_row_coords: array_like
                       Row indexes to be set to 1

    vertex_col_coords: array_like
                       Column indexes (following row indexes) set to 1

    shape: tuple

    Returns
    -------

    mask: np.uint8
          Boolean masked array
    """

    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords,
                                                    vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


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
        edgecolor = '#31a354'
    elif edgecolor == 'tumor':
        edgecolor = '#ca0020'  #'#f03b20'

    polygon = polygon - np.array([x0, y0])
    polygon = polygon * scale_factor
    polygon = np.round(polygon).astype(int)
    polygon = Polygon(polygon, edgecolor=edgecolor, facecolor=None, fill=False)
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
    for tumor_patch in tumor_patches:
        polygon = np.array(tumor_patch['vertices'])
        polygon = translate_and_scale_polygon(polygon, x0, y0, scale_factor,
                                              'tumor')
        polygons.append(polygon)
    for normal_patch in normal_patches:
        polygon = np.array(normal_patch['vertices'])
        polygon = translate_and_scale_polygon(polygon, x0, y0, scale_factor,
                                              'normal')
        polygons.append(polygon)
    if ax:
        for polygon in polygons:
            ax.add_patch(polygon)
    return polygons
