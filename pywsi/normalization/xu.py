import spams

from .macenko import MacenkoNormalization
from .color_conversion import RGB2OD, get_nonwhite_mask

# from sklearn.decomposition import NMF


class XuNormalization(MacenkoNormalization):
    def __init__(self, **kwargs):
        super(XuNormalization, self).__init__(**kwargs)

    def get_stain_matrix(self, source_image):
        """ OD = SV. Get V

        Parameters
        ----------
        source_image: array_like
                      np.unit8 array of rgb values

        Returns
        -------

        stain_matrix: array_like
                      2 x M matrix for a N x M matrix

        """
        OD = RGB2OD(source_image)
        OD = OD.reshape((-1, 3))
        if self.maskout_white:
            nonwhite_mask = get_nonwhite_mask(
                source_image, self.nonwhite_threshold
            ).reshape((-1,))
            OD = OD[nonwhite_mask]
        OD = OD[(OD > self.beta).any(axis=1), :]
        self.OD = OD
        """
        Objective ||X-Da||_2^ + lambda1 * ||a||_1 + lambda2 ||a||_2^2 +
        """
        param = {
            "K": 2,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "gamma1": self.gamma1,
            "mode": 2,
            "modeD": 1,
            "posD": True,
            "posAlpha": True,
            "verbose": False,
        }
        stain_matrix = spams.trainDL(OD.T, **param).T
        if stain_matrix[0, 0] < stain_matrix[1, 0]:
            stain_matrix = stain_matrix[[1, 0], :]
        return stain_matrix
