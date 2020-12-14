import numpy as np
from sklearn import datasets

import matplotlib.pyplot as plt


from tomaster import tomato, tomato_img
from tomaster.tomato import normalize_clusters


def clusters_equal(a, b):
    return np.all(normalize_clusters(a) == normalize_clusters(b))


def test_circles():
    X, y = datasets.make_circles(n_samples=500, noise=0.01, random_state=1337)
    clusters = tomato(points=X, k=5, tau=5)
    assert clusters_equal(y, clusters)
    clusters, tau = tomato(points=X, k=5, n_clusters=2)
    assert clusters_equal(y, clusters)


def test_moons():
    X, y = datasets.make_moons(n_samples=1000, noise=0.05, random_state=1337)
    clusters = tomato(points=X, k=5, tau=5)
    assert clusters_equal(y, clusters)
    clusters, tau = tomato(points=X, k=5, n_clusters=2)
    assert clusters_equal(y, clusters)


def test_blobs():
    X, y = datasets.make_blobs(n_samples=1000, random_state=42)
    clusters = tomato(points=X, k=5, tau=5)
    assert clusters_equal(y, clusters)
    clusters, tau = tomato(points=X, k=5, n_clusters=3)
    assert clusters_equal(y, clusters)
