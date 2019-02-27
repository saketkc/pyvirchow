import numpy as np


def welford_update_M1(prev_avg, new_value, count):
    """Welford's updated mean.

    mu_{n} = mu_{n-1} + 1/n(x_n - mu_{n-1}).

    Parameters
    ----------
    prev_avg: array_like
              Vector of (count-1)th step averages
    new_value: array_like
               New incoming values vector
    count: int
            Current count (count starts from 1)

    Returns
    -------

    new_avg: array_like
             Updated average

    """
    delta = new_value - prev_avg
    new_avg = prev_avg + delta / count
    return new_avg


def welford_update_M2(prev_M2, new_value, prev_avg, new_avg):
    """Welford's updated variance.

    S_n = S_{n-1} + (x_n-mu_{n-1})(x_n-mu_n)


    Parameters
    ----------
    prev_M2: array_like
             Vector of (count-1)th step M2 stats
    new_value: array_like
               New incoming values vector
    prev_avg: array_like
              Vector of (count-1)th step averages
    count: int
           Current count (count starts from 1)

    Returns
    -------
    new_M2: array_like
            Updated M2 stats
    """
    prev_avg = np.array(prev_avg)
    new_avg = np.array(new_avg)

    delta1 = new_value - prev_avg
    delta2 = new_value - new_avg
    new_M2 = prev_M2 + delta1 * delta2
    return new_M2


def welford_simulatenous_update(prev_avg, prev_M2, new_value, count):
    """Perform Welford's simulatenous update on mean and variance

    M2 = n*sigma_n^2

    Parameters
    ----------
    prev_avg: array_like
              Vector of (count-1)th step averages
    prev_M2: array_like
             Vector of (count-1)th step M2 stats
    new_value: array_like
               New incoming values vector
    count: int
           Current count (count starts from 1)

    Returns
    -------
    new_avg: array_like
             Updated average
    new_M2: array_like
            Updated M2 stats
    new_var: array_like
             Population variance
    new_samplevar: array_like
                   Sample Variance


    Example
    -------
    vector_matrix = [[1,2,3], [3,2,1], [5,1,3]]# [3,4,1]]

    for idx, vector in enumerate(vector_matrix):
        count = idx+1
        vector = np.array(vector)
        if count == 1:
            new_avg = vector
            prev_M2 = np.zeros(len(vector))

        avg, M2, new_var, new_samplevar = welford_simulatenous_update(avg, M2,
                                                                      vector, count)
        if count==2:
            M2 = 2*np.nanvar(np.array(vector_matrix[0:2]), axis=0)

    """
    if count == 1:
        new_avg = new_value
        new_M2 = np.zeros(len(new_value))
        return new_avg, new_M2, np.nan, np.nan
    new_avg = welford_update_M1(prev_avg, new_value, count)
    new_M2 = welford_update_M2(prev_M2, new_value, prev_avg, new_avg)
    if count > 1:
        return new_avg, new_M2, new_M2 / count, new_M2 / (count - 1)
    else:
        return new_avg, new_M2, np.nan, np.nan
