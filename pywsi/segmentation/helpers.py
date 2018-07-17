import numpy as np
import scipy.ndimage.measurements as ms


def collapse_labels(labelled_image):
    """Collapse labels in a labeleled image
    so that all labels are contigous

    Parameters
    ----------
    labelled_image: array_like
                    An image with labels

    Returns
    -------
    label_collapsed_image: array_like
                     Image with contigous labels
    """
    label_collapsed_image = labelled_image.copy()
    positions = ms.find_objects(labelled_image)
    index = 1
    for i in np.arange(1, len(positions) + 1):
        if positions[i - 1] is not None:
            # get patch
            patch = label_collapsed_image[positions[i - 1]]
            patch[patch == i] = index
            index += 1
    return label_collapsed_image


def collapse_small_area(labelled_image, minimum_area):
    """Collapse labelled image removing areas with too low are.


    Parameters
    ----------
    labelled_image: array_like
                    An image with labels
    minimum_area: float
                  Areas with this and above area are retained
    Returns
    -------
    label_collapsed_image: array_like
                     Image with contigous labels

    """
    collapsed_image = labelled_image.copy()
    collapsed_image = collapse_labels(collapsed_image)

    pixel_count, edges = np.histogram(
        collapsed_image, bins=collapsed_image.max() + 1)
    positions = ms.find_objects(collapsed_image)
    for i in range(1, pixel_count.size):
        if pixel_count[i] < minimum_area:
            patch = collapsed_image[positions[i - 1]]
            # Blacken out that patch
            patch[patch == i] = 0
    collapsed_image = collapse_labels(collapsed_image)
    return collapsed_image
