from functools import lru_cache

import numpy as np
from numba import njit
from sklearn.neighbors import NearestNeighbors

njit = njit(cache=True)


@njit
def _find(forest, i):
    if forest[i] != i:
        forest[i] = _find(forest, forest[i])
    return forest[i]


@njit
def _union(forest, a, b):
    forest[_find(forest, b)] = forest[_find(forest, a)]


@njit
def _tomato_pre(density, neighbors):
    n = len(density)
    forest = np.arange(n)

    ind = density.argsort()[::-1]
    order = ind.argsort()

    for i in ind:
        for j in neighbors[i]:
            if order[j] > order[i]:
                continue
            forest[i] = j

    return forest, order, ind


@njit
def _tomato(density, neighbors, tau, forest, order, ind):
    forest = forest.copy()

    for i in ind:
        if neighbors.shape[1] == 1 or tau == 0:
            continue
        for j in neighbors[i]:
            if order[j] > order[i]:
                continue
            ri, rj = _find(forest, i), _find(forest, j)
            if ri != rj and min(density[ri], density[rj]) < density[i] + tau:
                if order[ri] < order[rj]:
                    _union(forest, ri, rj)
                else:
                    _union(forest, rj, ri)

    for i in range(len(density)):
        _find(forest, i)
    return forest


def normalize_clusters(y):
    _, index, inverse = np.unique(y, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(index))
    return order[inverse]


def tomato(
    *,
    points=None,
    k=None,
    neighbors=None,
    distances=None,
    tau=None,
    n_clusters=None,
    relative_tau: bool = True,
    keep_cluster_labels: bool = False,
):
    """ToMATo clustering

    Parameters
    ----------

    points : np.ndarray
        Array of shape (n, dim)
    k : int
        Number of nearest neighbors to build the graph with
    neighbors : np.ndarray
        Array of shape (n, dim)
    distances : np.ndarray
        Array of shape (n, dim)
    tau : float or None
        Prominence threshold. Must not be specified if `n_clusters` is given.
    relative_tau : bool
        If `relative_tau` is set to `True`, `tau` will be multiplied by the standard deviation of the densities, making easier to have a unique value of `tau` for multiple datasets.
    n_clusters : int or None
        Target number of clusters. Must not be specified if `tau` is given.
    keep_cluster_labels : bool
        If False, converts the labels to make them contiguous and start from 0.

    Returns
    -------

    clusters : np.ndarray
        Array of shape (n,) containing the cluster indexes.
    tau : float
        Prominence threshold. Only present if `n_clusters` was given.

    """

    assert [tau, n_clusters].count(
        None
    ) == 1, "You cannot give both `tau` and `n_clusters`"
    assert n_clusters is None or n_clusters > 0

    assert (points is None) == (k is None)
    assert (neighbors is None) == (distances is None)
    assert (points is not None) or (neighbors is not None)
    if neighbors is None:
        distances, neighbors = NearestNeighbors(n_neighbors=k).fit(points).kneighbors()
    density = ((distances ** 2).mean(axis=-1) + 1e-10) ** -0.5
    pre = _tomato_pre(density, neighbors)

    if tau is not None:
        if relative_tau:
            tau *= density.std()
        ans = _tomato(density, neighbors, tau, *pre)
    else:

        @lru_cache(1)
        def aux1(tau):
            return _tomato(density, neighbors, np.float32(tau), *pre)

        def aux2(tau):
            return len(np.unique(aux1(tau)))

        if aux2(0) < n_clusters:
            # error
            tau = -1
            ans = aux1(0)
        else:
            a = 0
            b = density.max() - density.min() + 1

            if aux2(b) > n_clusters:
                # error
                tau = -1
                ans = aux1(b)
            else:
                # binary search
                while aux2((a + b) / 2) != n_clusters:
                    print(a, b, aux2((a + b) / 2))
                    if aux2((a + b) / 2) > n_clusters:
                        a = (a + b) / 2
                    else:
                        b = (a + b) / 2

            tau = (a + b) / 2
            ans = aux1(tau)

    if not keep_cluster_labels:
        ans = normalize_clusters(ans)

    if n_clusters is None:
        return ans
    else:
        return ans, tau


def tomato_img(
    img: np.ndarray, *, spatial_weight: float = 0, lab_space: bool = True, **kwargs
):
    """ToMATo for images

    Parameters
    ----------

    img : np.ndarray
        Image of shape (h, w) or (h, w, 3)
    spatial_weight : float
        Importance of the pixel positions in the distance function
    lab_space : bool
        If True, converts color images to the CIE L*a*b color space (<https://en.wikipedia.org/wiki/CIELAB_color_space>)

    see tomato() for other arguments.

    Returns
    -------

    clusters : np.ndarray
        Array of shape (h, w) containing the cluster indexes.
    """
    from skimage.color import rgb2lab
    from skimage.util import img_as_float

    assert len(img.shape) in [2, 3]
    if len(img.shape) == 3:
        assert img.shape[2] in [1, 3]

    img = img_as_float(img)

    if len(img.shape) == 3 and lab_space:
        img = rgb2lab(img)
    else:
        img = img[:, :, None]
        img *= 100

    ndims = img.shape[-1]
    coords = np.indices(img.shape[:2], dtype=np.float32).reshape(2, -1).T
    coords *= spatial_weight
    points = np.concatenate((coords, img.reshape(-1, ndims)), 1)
    ans = tomato(points=points, **kwargs)
    if isinstance(ans, tuple):
        ans = ans[0]
    return ans.reshape(img.shape[:2])
