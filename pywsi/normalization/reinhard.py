class ReinhardNormalization(object):
    """Reinhard Normalization """

    def __init__(self):
        self.target_l = None
        self.target_a = None
        self.target_b = None
        self.target_mean = (None, None, None)
        self.target_std = (None, None, None)

    def read_tiff(self, target_image):
        target_image = OpenSlide(target_image)
        level_used = target_image.level_count - 1
        rgb_image = np.array(
            target_image.read_region(
                (0, 0), level_used,
                target_image.level_dimensions[level_used]).convert('RGB'))
        target_image = cv2.cvtColor(rgb_image,
                                    cv2.COLOR_RGB2LAB).astype(np.float32)
        return target_image

    def fit(self, target_image):
        if isinstance(target_image, str):
            if 'tif' in target_image:
                target_image = self.read_tiff(target_image)
            else:
                target_image = cv2.imread(target_image)
                target_image = cv2.cvtColor(
                    target_image, cv2.COLOR_BGR2LAB).astype(np.float32)

        self.target_l, self.target_a, self.target_b = cv2.split(target_image)
        self.target_mean, self.target_std = self.get_mean_and_std(target_image)
        return cv2.cvtColor(target_image.astype(np.uint8), cv2.COLOR_LAB2RGB)

    def transform(self, source_image):
        if isinstance(source_image, str):
            if 'tif' in source_image:
                source_image = self.read_tiff(source_image)
            else:
                source_image = cv2.imread(source_image)
                source_image = cv2.cvtColor(
                    source_image, cv2.COLOR_BGR2LAB).astype(np.float32)

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

        transfer = cv2.cvtColor(transfer.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return transfer

    @staticmethod
    def get_mean_and_std(image):
        """Get image mean and std for all channels
        """
        c1, c2, c3 = cv2.split(image)
        return (c1.mean(), c2.mean(), c3.mean()), (c1.std(), c2.std(),
                                                   c3.std())
