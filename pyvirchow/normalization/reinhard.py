import numpy as np
from ..io import read_as_lab
from skimage.color import lab2rgb


class ReinhardNormalization(object):
    """Reinhard Normalization: https://ieeexplore.ieee.org/document/946629 """

    def __init__(self):
        self.target_l = None
        self.target_a = None
        self.target_b = None
        self.target_mean = (None, None, None)
        self.target_std = (None, None, None)

    def fit(self, target_image):
        """Fit target image by storing its mean/std.

        Parameters
        ----------
        target_image: array_like
                      np.uint8 of rgb values

        """
        target_image = read_as_lab(target_image)
        self.target_l, self.target_a, self.target_b = (
            target_image[:, :, 1],
            target_image[:, :, 1],
            target_image[:, :, 2],
        )
        self.target_mean, self.target_std = self.get_mean_and_std(target_image)

    def transform(self, source_image):
        """Perform normalization on source image.

        Parameters
        ----------
        source_image: array_like
                      np.uint8 of rgb values

        """
        source_image = read_as_lab(source_image)
        source_l, source_a, source_b = (
            source_image[:, :, 0],
            source_image[:, :, 1],
            source_image[:, :, 2],
        )
        source_mean, source_std = self.get_mean_and_std(source_image)

        source_l -= source_mean[0]
        source_a -= source_mean[1]
        source_b -= source_mean[2]

        source_l = self.target_std[0] / source_std[0] * source_l
        source_a = self.target_std[1] / source_std[1] * source_a
        source_b = self.target_std[2] / source_std[2] * source_b

        source_l += self.target_mean[0]
        source_a += self.target_mean[1]
        source_b += self.target_mean[2]

        source_l = np.clip(source_l, 0, 255)
        source_a = np.clip(source_a, 0, 255)
        source_b = np.clip(source_b, 0, 255)

        transfer = np.dstack((source_l, source_a, source_b))
        transfer = lab2rgb(transfer)
        return transfer

    @staticmethod
    def get_mean_and_std(image):
        """Get image mean and std for all channels
        """
        c1, c2, c3 = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        return (c1.mean(), c2.mean(), c3.mean()), (c1.std(), c2.std(), c3.std())
