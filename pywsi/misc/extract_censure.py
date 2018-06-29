from skimage import feature
import numpy as np
from skimage.color import rgb2gray
from ..io.operations import read_as_rgb


def _censure_features(image_path, savelocation):
    """Extract and store CENSURE features from image.

    Parameters
    ----------
    path: string
          Location to image file

    """
    image = rgb2gray(read_as_rgb(image_path))
    censure = feature.CENSURE()
    censure.detect(image)
    keypoints = censure.keypoints
    return keypoints
