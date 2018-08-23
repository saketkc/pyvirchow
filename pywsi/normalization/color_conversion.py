import numpy as np
from skimage.color import rgb2lab
from skimage.color import lab2rgb
from numpy import linalg as LA


def RGB2OD(image):
    """Convert Intensities to Optical Density"""
    assert np.issubdtype(image.dtype, np.uint8)
    image[np.where(image <= 0)] = 1
    return (-np.log(image / 255.0))


def OD2RGB(OD):
    """Convert optical density back to RGB"""
    return 255 * np.exp(-OD)


def normalize_brightness(rgb_image, percentile=50):
    """Correct image brightness

    Transforms the image in Lab space where
     L = brightness
     a = green-red
     b = blue-yellow

     This can be acheived by normalizing based on say 50th percentile of L
     values and then converting back to rgb

    Parameters
    ----------
    rgb_image: array_like
               np.uint8 RGB image


    """
    lab_image = rgb2lab(rgb_image)
    L = lab_image[:, :, 0]
    L_p = np.percentile(L, percentile)
    L_scaled = L / L_p * 255
    L_normalized = np.clip(L_scaled, 0, 255)
    lab_image[:, :, 0] = L_normalized.astype(np.uint8)
    return lab2rgb(lab_image)


def get_nonwhite_mask(rgb_image, threshold=0.8):
    """Get a mask of non-white spots from an image.

    Parameters
    ----------
    rgb_image: array_like
               Input

    threshold: float
               What leel of intensity (wrt 255) should the values be thresholded

    Returns
    -------
    nonwhite_mask: array_like
                   A binary array with 0s at all close to white spots
    """
    lab_image = rgb2lab(rgb_image)
    L = lab_image[:, :, 0]
    L_scaled = L / 255.0
    return (L_scaled < threshold)


def normalize_rows(X, ord=1):
    """Normalize row by its 1/2/inf norm

    Parameters
    ----------
    X: array_like
       Input

    ord: int
         order of norm

    Returns
    -------
    normalized_X: array_like
                  Normalized X, normalized by row norm
    """
    return X / LA.norm(X, axis=1, ord=ord)[:, np.newaxis]
