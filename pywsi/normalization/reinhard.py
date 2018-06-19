from ..io import rgb_to_lab
from ..io import lab_to_rgb


class ReinhardNormalization(object):
    """Reinhard Normalization """

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
        target_image = lab_to_rgb(target_image)
        self.target_l, self.target_a, self.target_b = cv2.split(target_image)
        self.target_mean, self.target_std = self.get_mean_and_std(target_image)

    def transform(self, source_image):
        source_l, source_a, source_b = cv2.split(source_image)
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

        transfer = cv2.merge([source_l, source_a, source_b])
        transfer = lab_to_rgb(transfer)
        return transfer

    @staticmethod
    def get_mean_and_std(image):
        """Get image mean and std for all channels
        """
        c1, c2, c3 = cv2.split(image)
        return (c1.mean(), c2.mean(), c3.mean()), (c1.std(), c2.std(),
                                                   c3.std())
