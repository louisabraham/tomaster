#!/usr/bin/env python3

import os
from setuptools import setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()


setup(
    name="tomaster",
    version="0.0.4",
    author="Louis Abraham",
    license="MIT",
    author_email="louis.abraham@yahoo.fr",
    description="Topological Mode Analysis on Steroids",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/louisabraham/tomaster",
    packages=["tomaster"],
    install_requires=["numpy", "numba", "scikit-learn", "scikit-image"],
    tests_require=["pytest"],
    classifiers=[
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ],
)
