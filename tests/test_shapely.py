from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest

from pywsi.morphology.mask import get_common_interior_polygons
from shapely.geometry import Polygon, box
from pywsi.io.operations import poly2mask
import numpy as np
import numpy.testing as npt


def test_common_interior():
    p1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    p2 = Polygon([(0, 0), (3, 0), (2, 2), (0, 2)])
    p3 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    common = get_common_interior_polygons(p1, [p2, p3])
    assert common == [0]


def test_mask_creation():
    p1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    p2 = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])

    # Define a filled mask
    # of size (4, 4) which is all filled
    mask_full = poly2mask([p2], (4, 4))

    mask_inner = poly2mask([p1], (4, 4))

    # Subtract a 3x3 rectangle from 4x4 rectangle
    """
     ____
    |    |
    |__  |
    |  | |
    |__|_|

    """
    mask_sub = mask_full - mask_inner

    # We are in array domain, so the
    npt.assert_equal(
        mask_sub,
        np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 1, 1]]))
