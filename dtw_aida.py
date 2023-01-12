from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from math import isinf
import numpy as np
import math

def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


def accelerated_dtw(x, y, dist, warp=1):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)

    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    :param int warp: how many shifts are computed.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    if ndim(x) == 1:
        x = x.reshape(-1, 1)
    if ndim(y) == 1:
        y = y.reshape(-1, 1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:, 1:] = cdist(x, y, dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j],
                             D0[i, min(j + k, c)]]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


def balanced_kl_divergence(p, q):
    np.seterr(divide='ignore', invalid='ignore')
    kl1_abs = abs(np.where(p != 0, p * np.log(p / q), 0))
    filtered_kl1 = [v for v in kl1_abs if not (math.isinf(v) or math.isnan(v))]
    kl2_abs = abs(np.where(q != 0, q * np.log(q / p), 0))
    filtered_kl2 = [v for v in kl2_abs if not (math.isinf(v) or math.isnan(v))]
    kl1 = np.sum(filtered_kl1)
    kl2 = np.sum(filtered_kl2)
    kl_balanced = (kl1 + kl2) / 2
    return kl_balanced


def soma_diagonal(mat, n):
    principal = 0
    for i in range(0, n):
        principal += mat[i][i]
    return principal

def create_histogram_list(df1, df2):
    if (df1.shape[0] > 0 and df2.shape[0] > 0):
        gk1 = df1.groupby(['10'])
        gk2 = df2.groupby(['10'])

        dataframes = [group for _, group in gk1]
        dataframes2 = [group for _, group in gk2]
        histograms_list = dataframes + dataframes2

        return histograms_list

def compute_histogram_dtw(histograms_list):
    list_dtw_k = []
    for hist1 in histograms_list:
        for hist2 in histograms_list:
            if (hist1 is not None and hist2 is not None):
                hist_df1 = hist1.iloc[:, :10]
                hist_df2 = hist2.iloc[:, :10]
                p_probs = np.array(hist_df1)
                q_probs = np.array(hist_df2)
                dtw_k, cost_matrix_k, acc_cost_matrix_k, path_k = dtw(p_probs, q_probs,
                                                                      dist=balanced_kl_divergence)
                # Sum diagonal items
                euclidian = soma_diagonal(cost_matrix_k.T, min(cost_matrix_k.T.shape) - 1)
                # Create dataframe using Kullback Leibor distance
                row_dtw_k = [hist1.iloc[0, 10], hist2.iloc[0, 10], dtw_k, euclidian, path_k, cost_matrix_k.T,
                             len(np.unique(path_k[0])),
                             len(np.unique(path_k[1])), len(path_k[0]), min(cost_matrix_k.T.shape)]
                list_dtw_k.append(row_dtw_k)
    return list_dtw_k

def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)
