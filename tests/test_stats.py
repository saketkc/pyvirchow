import pytest
from pywsi.stats import welford_simulatenous_update
import numpy as np
import numpy.testing as npt


def test_welford():
    vector_matrix = [[1,2,3], [3,2,1], [5,1,3], [6, 7, 8]]

    for idx, vector in enumerate(vector_matrix):
        count = idx+1
        vector = np.array(vector)
        if count == 1:
            avg = vector
            M2 = np.zeros(len(vector))

        avg, M2, new_var, new_samplevar = welford_simulatenous_update(avg, M2,
                                                                      vector, count)
        if count==2:
            M2 = 2*np.nanvar(np.array(vector_matrix[0:2]), axis=0)
    npt.assert_almost_equal(new_var, np.nanvar(np.array(vector_matrix), axis=0))
    npt.assert_almost_equal(avg, np.nanmean(np.array(vector_matrix), axis=0))

if __name__ == '__main__':
    pytest.main([__file__])
