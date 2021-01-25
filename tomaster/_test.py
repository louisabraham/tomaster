import numpy as np
from sklearn import datasets

from tomaster import tomato, tomato_img


def normalize_clusters(y):
    _, index, inverse = np.unique(y, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(index))
    return order[inverse]


def clusters_equal(a, b):
    return np.all(normalize_clusters(a) == normalize_clusters(b))


def test_circles():
    X, y = datasets.make_circles(n_samples=500, noise=0.01, random_state=1337)
    clusters = tomato(points=X, k=5, n_clusters=2)
    assert clusters_equal(y, clusters)


def test_moons():
    X, y = datasets.make_moons(n_samples=1000, noise=0.05, random_state=1337)
    clusters = tomato(points=X, k=5, n_clusters=2)
    assert clusters_equal(y, clusters)


def test_blobs():
    X, y = datasets.make_blobs(n_samples=1000, random_state=42)
    clusters = tomato(points=X, k=5, n_clusters=3)
    assert clusters_equal(y, clusters)

