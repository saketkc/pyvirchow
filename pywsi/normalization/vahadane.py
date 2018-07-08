from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import spams

from .macenko import RGB2OD, MacenkoNormalization


class VahadaneNormalization(MacenkoNormalization):
    def __init__(self, **kwargs):
        super(VahadaneNormalization, self).__init__(**kwargs)

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
        OD = (OD[(OD > self.beta).any(axis=1), :])
        self.OD = OD
        param = {
            'K': 2,
            'lambda1': self.lambda1,
            'mode': 2,
            'modeD': 0,
            'posD': True,
            'posAlpha': True,
            'verbose': False
        }
        stain_matrix = spams.trainDL(OD.T, **param).T
        if stain_matrix[0, 0] < stain_matrix[1, 0]:
            stain_matrix = stain_matrix[[1, 0], :]
        return stain_matrix
