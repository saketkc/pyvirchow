from skimage import draw
import numpy as np


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
