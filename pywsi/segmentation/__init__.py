from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
import matplotlib
matplotlib.use('Agg', warn=False)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .diff_gaussian import laplace_of_gaussian
from .graph_cut import perform_binary_cut
from .helpers import collapse_labels
from .helpers import collapse_small_area
from .binary_segmentation import poisson_deconvolve
from .binary_segmentation import gmm_thresholding

import numpy as np
import scipy as sp
from skimage.color import label2rgb, rgb2gray
from skimage.measure import regionprops

from .max_clustering import max_clustering
from .fractal_dimension import fractal_dimension
import pandas as pd
import warnings

__all__ = ('max_clustering', )


def label_nuclei(nuclei_stain_rgb,
                 thresholding='gmm',
                 foreground_threshold=None,
                 min_radius=3,
                 max_radius=11,
                 local_max_search_radius=3,
                 min_nucleus_area=80,
                 draw=True):
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

    region_properties = regionprops(
        nuclei_seg_mask, intensity_image=nuclei_stain_bw, coordinates='rc')

    title = 'Number of nuclei = {}'.format(len(region_properties))

    # Display results
    if not draw:
        return region_properties, foreground_mask
    fig = plt.figure(figsize=(20, 10))

    ax = plt.subplot(1, 2, 1)
    ax.set_axis_off()
    ax.imshow(
        label2rgb(nuclei_seg_mask, nuclei_stain_bw, bg_label=0),
        origin='lower')
    ax.set_title(
        'Nuclei segmentation mask overlay \n {}'.format(title), fontsize=16)

    ax = plt.subplot(1, 2, 2)
    ax.imshow(nuclei_stain_rgb)
    ax.set_axis_off()
    ax.set_xlim([0, nuclei_stain_rgb.shape[1]])
    ax.set_ylim([0, nuclei_stain_rgb.shape[0]])
    ax.set_title('Nuclei bounding boxes', fontsize=16)

    for region_property in region_properties:

        c = [region_property.centroid[1], region_property.centroid[0], 0]
        width = region_property.bbox[3] - region_property.bbox[1] + 1
        height = region_property.bbox[2] - region_property.bbox[0] + 1

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
    return region_properties, foreground_mask


def extract_features(region_property):
    """Given a region_property extract important features.

    Parameters
    ----------
    region_property: skimage.measre.RegionProperties

    Returns
    -------
    features: dict
              Dictionary with feature id as key
    """
    features_to_keep = [
        'area', 'bbox_area', 'convex_area', 'eccentricity',
        'equivalent_diameter', 'extent', 'inertia_tensor_eigvals',
        'major_axis_length', 'minor_axis_length', 'max_intensity',
        'mean_intensity', 'moments_central', 'moments_hu', 'orientation',
        'perimeter', 'solidity'
    ]
    features = OrderedDict()
    for feature in features_to_keep:
        if feature in [
                'inertia_tensor_eigvals', 'moments_hu', 'moments_central'
        ]:
            moments = region_property[feature]
            moments = np.asarray(moments)
            for index, moment in enumerate(moments.ravel()):
                features['{}_{}'.format(feature, index + 1)] = moment
        else:
            try:
                features[feature] = region_property[feature]
            except KeyError:
                features[feature] = np.nan

    # Custom features
    # compactness = perimeter^2/area
    features['compactness'] = features['perimeter']**2 / features['area']
    # Texture
    intensity = region_property['intensity_image']
    features['texture'] = np.var(intensity)
    features['fractal_dimension'] = fractal_dimension(intensity)
    return features


def summarize_region_properties(region_properties, image):
    """Summarize RegionProperties over an entire image.

    Parameters
    ----------
    region_properties: list
                       List of region propeties

    Returns
    -------
    summary_stats: dict
                   Summarized info
    """
    feature_df = []
    for region_property in region_properties:
        feature_df.append(extract_features(region_property))
    feature_df = pd.DataFrame.from_dict(feature_df)
    if len(feature_df.index) == 0:
        return None
    features = feature_df.mean(skipna=True).to_dict()
    features['total_nuclei_area'] = float(feature_df[['area']].sum())
    features['total_nuclei_area_ratio'] = float(
        feature_df[['area']].sum()) / (image.shape[0] * image.shape[1])
    features['nuclei'] = len(feature_df.index)
    features['mean_intensity_entire_image'] = np.mean(rgb2gray(image))
    features[
        'nuclei_intensity_over_entire_image'] = features['mean_intensity'] / np.mean(
            rgb2gray(image))

    return features
