# coding: utf-8

"""
Package setup file.
"""

import os
from setuptools import setup, find_packages


this_dir = os.path.dirname(os.path.abspath(__file__))


# package keyworkds
keywords = [
    "CERN", "CMS", "LHC", "machine learning", "tensorflow", "keras",
]


# package classifiers
classifiers = [
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Development Status :: 4 - Beta",
    "Operating System :: OS Independent",
    "License :: OSI Approved :: BSD License",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Information Technology",
]


# helper to read non-empty, stripped lines from an opened file
def readlines(f):
    for line in f.readlines():
        if line.strip():
            yield line.strip()


# read the readme file
with open(os.path.join(this_dir, "README.rst"), "r") as f:
    long_description = f.read()


# load installation requirements
with open(os.path.join(this_dir, "requirements.txt"), "r") as f:
    install_requires = list(readlines(f))


# load docs requirements
with open(os.path.join(this_dir, "requirements_docs.txt"), "r") as f:
    docs_requires = [line for line in readlines(f) if line not in install_requires]


# load meta infos
meta = {}
with open(os.path.join(this_dir, "cmsml", "__meta__.py"), "r") as f:
    exec(f.read(), meta)


# install options
options = {}


setup(
    name="cmsml",
    version=meta["__version__"],
    author=meta["__author__"],
    author_email=meta["__email__"],
    description=meta["__doc__"].strip().split("\n")[0].strip(),
    license=meta["__license__"],
    url=meta["__contact__"],
    keywords=" ".join(keywords),
    classifiers=classifiers,
    long_description=long_description,
    install_requires=install_requires,
    python_requires=">=3.6, <=3.11'",
    extras_require={
        "docs": docs_requires,
    },
    entry_points={
        "console_scripts": [
            "cmsml_open_tf_graph = cmsml.scripts.open_tf_graph:main",
        ],
    },
    zip_safe=False,
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    options=options,
)
