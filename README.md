[![PyPI
version](https://badge.fury.io/py/tomaster.svg)](https://badge.fury.io/py/tomaster)
[![Downloads](https://pepy.tech/badge/tomaster)](https://pepy.tech/project/tomaster)

[![Build
Status](https://travis-ci.org/louisabraham/tomaster.svg?branch=master)](https://travis-ci.org/louisabraham/tomaster)

# tomaster: Topological Mode Analysis on Steroids

`tomaster` implements algorithms for topological mode analysis.

The code is simple to read because it is written in pure Python.

The performance is good thanks to jit compilation with
[numba](https://numba.pydata.org/).

# Usage

``` pycon
>>> from tomaster import tomato
>>> from sklearn import datasets
>>> X, y = datasets.make_moons(n_samples=1000, noise=0.05, random_state=1337)
>>> clusters = tomato(points=X, k=5, n_clusters=2)

>>> import matplotlib.pyplot as plt
>>> plt.scatter(*X.T, c=clusters)
>>> plt.show()
```

![](https://raw.githubusercontent.com/louisabraham/tomaster/master/examples/moons.png)

# Installation

    pip install tomaster

# Testing

    pytest

# API

``` python
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
```

# References

  - Chazal, Frédéric, Leonidas J. Guibas, Steve Y. Oudot, and Primoz
    Skraba. "Persistence-based clustering in riemannian manifolds."
    Journal of the ACM (JACM) 60, no. 6 (2013): 41.
    [\[pdf\]](https://geometrica.saclay.inria.fr/data/Steve.Oudot/clustering/jacm_oudot.pdf)

  - Reference implementations:
    <https://geometrica.saclay.inria.fr/data/ToMATo/>,
    <https://geometrica.saclay.inria.fr/data/Steve.Oudot/clustering/>
