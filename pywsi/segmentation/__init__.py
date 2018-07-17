from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib
matplotlib.use('Agg', warn=False)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .diff_gaussian import laplace_of_gaussian
from .graph_cut import perform_binary_cut
from .helpers import collapse_labels
from .helpers import collapse_small_area
from .max_clustering import max_clustering
from .binary_segmentation import poisson_deconvolve
from .binary_segmentation import gmm_thresholding

import numpy as np
import scipy as sp
from skimage.color import label2rgb, rgb2gray
from skimage.measure import regionprops


def label_nuclei(nuclei_stain_rgb,
                 thresholding='gmm',
                 foreground_threshold=None,
                 min_radius=3,
                 max_radius=11,
                 local_max_search_radius=3,
                 min_nucleus_area=80):
    """Perform segmentation labelling on nuclei.

    Parameters
    ----------
    nuclei_stain: array_like
                  Gray image matrix
    thresholding: string
                  Choice of binary/gmm/poisson
    min_radius, max_radius: int
                            Minimum/maximum radius of detectable bob by difference of gaussian
    local_max_search_radius: int
                             how many neraby regions should be considered for collapsing into one
    min_nucleus_area: float
                      Anything below this is not a viable Nuclei

    """
    assert thresholding in [
        'gmm', 'poisson', 'custom'
    ], 'Unsupported thresholding method {}'.format(thresholding)
    nuclei_stain_bw = rgb2gray(nuclei_stain_rgb)
    if thresholding == 'custom':
        assert foreground_threshold > 0, 'foreground_threshold should be > 0 for custom thresholding'
        foreground_mask = sp.ndimage.morphology.binary_fill_holes(
            nuclei_stain_bw < foreground_threshold)
    elif thresholding == 'gmm':
        foreground_threshold, _ = gmm_thresholding(nuclei_stain_bw)
        foreground_mask = sp.ndimage.morphology.binary_fill_holes(
            nuclei_stain_bw < foreground_threshold)
    elif thresholding == 'poisson':
        bg, fg, threshold = poisson_deconvolve(
            nuclei_stain_bw.astype(np.uint8))
        foreground_mask = perform_binary_cut(background=bg, foreground=fg)

    log_max, sigma_max = laplace_of_gaussian(
        nuclei_stain_bw,
        foreground_mask,
        sigma_min=min_radius * np.sqrt(2),
        sigma_max=max_radius * np.sqrt(2))

    nuclei_seg_mask, seeds, maxima = max_clustering(log_max, foreground_mask,
                                                    local_max_search_radius)

    # Remove small objects
    nuclei_seg_mask = collapse_small_area(nuclei_seg_mask,
                                          min_nucleus_area).astype(np.int)

    region_properties = regionprops(nuclei_seg_mask)

    print('Number of nuclei = ', len(region_properties))

    # Display results
    fig = plt.figure(figsize=(20, 10))

    ax = plt.subplot(1, 2, 1)
    ax.imshow(
        label2rgb(nuclei_seg_mask, nuclei_stain_bw, bg_label=0),
        origin='lower')
    ax.set_title('Nuclei segmentation mask overlay', fontsize=12)

    ax = plt.subplot(1, 2, 2)
    ax.imshow(nuclei_stain_rgb)
    ax.set_xlim([0, nuclei_stain_rgb.shape[1]])
    ax.set_ylim([0, nuclei_stain_rgb.shape[0]])
    ax.set_title('Nuclei bounding boxes', fontsize=12)

    for i in range(len(region_properties)):

        c = [
            region_properties[i].centroid[1], region_properties[i].centroid[0],
            0
        ]
        width = region_properties[i].bbox[3] - region_properties[i].bbox[1] + 1
        height = region_properties[i].bbox[2] - region_properties[i].bbox[0] + 1

        cur_bbox = {
            'type': 'rectangle',
            'center': c,
            'width': width,
            'height': height,
        }

        ax.plot(c[0], c[1], 'g+')
        mrect = mpatches.Rectangle(
            [c[0] - 0.5 * width, c[1] - 0.5 * height],
            width,
            height,
            fill=False,
            ec='g',
            linewidth=2)
        ax.add_patch(mrect)
    fig.tight_layout()
    plt.axis('off')
