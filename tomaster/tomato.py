import numpy as np

from numba import njit
from numba.typed import List, Dict

from sklearn.neighbors import NearestNeighbors

njit = njit(cache=True)


@njit
def _find(forest, i):
    if forest[i] != i:
        forest[i] = _find(forest, forest[i])
    return forest[i]


@njit
def raw_tomato(density, neighbors):
    """ToMATo clustering

    Parameters
    ----------
    density : np.ndarray
        array of n densities
    neighbors : np.ndarray
        array of shape (n, k) neighbors

    Returns
    -------
    edges : list of (float, int, int)
        list of merge edges with persistences
    """

    n = len(density)
    forest = np.empty(n, dtype=np.int64)

    edges = List()
    for i in range(n):
        edges.append((np.inf, i, i))

    ind = density.argsort()[::-1]
    order = ind.argsort()

    for i in ind:
        forest[i] = i
        for j in neighbors[i]:
            if order[j] < order[forest[i]]:
                forest[i] = j

        if forest[i] == i:
            continue
        edges[i] = (0.0, i, forest[i])
        ri = _find(forest, i)

        for j in neighbors[i]:
            if order[j] > order[i]:
                continue
            rj = _find(forest, j)
            if ri == rj:
                continue
            if order[ri] < order[rj]:
                edges[rj] = (density[rj] - density[i], i, j)
                forest[rj] = ri
            else:
                edges[ri] = (density[ri] - density[i], i, j)
                forest[ri] = ri = rj
    return edges


@njit
def clusters(edges, tau: float, keep_cluster_labels: bool = False):
    """Compute clusters from the output of raw_tomato

    Parameters
    ----------
    edges
        output of raw_tomato
    tau : float
        persistence gap
    keep_cluster_labels : bool
        if False, converts the labels to make them contiguous and start from 0

    Returns
    -------
    clusters : np.ndarray
        cluster identifiers
    """
    n = len(edges)
    forest = np.arange(n)
    for p, a, b in edges:
        if p <= tau:
            forest[_find(forest, a)] = _find(forest, b)
    for i in range(n):
        _find(forest, i)

    if not keep_cluster_labels:
        d = Dict.empty(
            key_type=np.int64,
            value_type=np.int64,
        )

        for x in forest:
            if x not in d:
                d[x] = len(d)
        for i in range(n):
            forest[i] = d[forest[i]]
    return forest


def tomato(
    *,
    points=None,
    k=None,
    neighbors=None,
    distances=None,
    density=None,
    metric="l2",
    bandwidth=None,
    raw: bool = False,
    tau=None,
    n_clusters=None,
    keep_cluster_labels: bool = False,
):
    """ToMATo clustering

    You can call this function with a lot of different signatures as it tries to build the missing parameters from the others.

    Parameters
    ----------

    points : np.ndarray
        Array of shape (n, dim)
    k : int
        Number of nearest neighbors to build the graph with
    neighbors : np.ndarray
        Array of shape (n, k)
    distances : np.ndarray
        Array of shape (n, k)
    density : np.ndarray
        Array of shape (n,)
    metric: str
        "l2" or "cosine"

    raw : bool
        if True, returns the merge edges

    tau : float
        Prominence threshold. If not specified, automatically selects the largest persistence gap.
    n_clusters : int
        Target number of clusters.

    keep_cluster_labels : bool
        If False, converts the labels to make them contiguous and start from 0.

    Returns
    -------
    clusters : np.ndarray
        if raw is False (default), array of shape (n,) containing the cluster indices
    edges : list
        if raw is True, spanning tree as list of (persistence, point1, point2)
    """

    assert metric in {"l2", "cosine"}

    def _points():
        assert points is not None

    def _neighbors():
        nonlocal distances, neighbors
        if neighbors is None:
            _points()
            assert k is not None
            distances, neighbors = (
                NearestNeighbors(n_neighbors=k).fit(points).kneighbors()
            )

    def _distances():
        nonlocal distances
        if distances is None:
            _points()
            _neighbors()
            if metric == "l2":
                a = points[:, None, :]
                b = points[neighbors.flatten()].reshape(*neighbors.shape, -1)
                distances = np.sqrt(np.sum((a - b) ** 2, -1))
            elif metric == "cosine":
                p = points / np.linalg.norm(points, axis=-1)
                a = p[:, None, :]
                b = p[neighbors.flatten()].reshape(*neighbors.shape, -1)
                distances = 1 - np.sum(a * b, axis=-1)

    def _density():
        nonlocal density
        if density is None:
            _distances()
            if metric == "l2":
                if bandwidth is None:
                    density = ((distances ** 2).mean(axis=-1) + 1e-10) ** -0.5
                else:
                    density = np.exp(-((distances / bandwidth) ** 2)).sum(axis=-1)
            elif metric == "cosine":
                assert bandwidth is not None, "bandwidth must be specified"
                assert bandwidth > 0
                density = np.exp(-((np.arccos(1 - distances) / bandwidth) ** 2)).sum(
                    axis=-1
                )

    _density()
    _neighbors()
    edges = raw_tomato(density, neighbors)

    if raw:
        return edges
    elif tau is not None:
        assert n_clusters is None
        return clusters(edges, tau, keep_cluster_labels)
    else:
        sp = sorted((p for p, _, _ in edges), reverse=True)
        if n_clusters is None:
            n_clusters = max(range(2, len(sp)), key=lambda i: sp[i - 1] - sp[i])
        tau = sp[n_clusters]
        return clusters(edges, tau, keep_cluster_labels)


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

