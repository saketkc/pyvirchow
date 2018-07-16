from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import maxflow

import gco


def perform_binary_cut(foreground, background):
    """Perform graph-cut based segmentation.

    Parameters
    ----------
    foreground, background: float32
                            Foreground, background probability matrix (w*h*3)



    """
    eps = np.finfo('float').eps
    foreground_matrix = -np.log(foreground + eps) / np.log(eps)
    background_matrix = -np.log(background + eps) / np.log(eps)
    stacked = np.dstack((foreground_matrix, background_matrix))
    pairwise = 1 - np.eye(2)
    print(stacked.shape, pairwise.shape)
    segmentation = gco.cut_grid_graph_simple(
        stacked, pairwise, n_iter=-1, algorithm='expansion')
    return segmentation.reshape(foreground.shape)
