# coding: utf-8

"""
Package setup file.
"""

import os
from setuptools import setup, find_packages


this_dir = os.path.dirname(os.path.abspath(__file__))


# package keyworkds
keywords = [
    "CERN", "CMS", "LHC", "machine learning",
]


# package classifiers
classifiers = [
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Development Status :: 4 - Beta",
    "Operating System :: OS Independent",
    "License :: OSI Approved :: BSD License",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Information Technology",
]


# read the readme file
with open(os.path.join(this_dir, "README.rst"), "r") as f:
    long_description = f.read()


# load installation requirements
with open(os.path.join(this_dir, "requirements.txt"), "r") as f:
    install_requires = [line.strip() for line in f.readlines() if line.strip()]


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
    python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, !=3.5.*, <4'",
    extras_require={
        "docs": ["sphinx", "sphinx-rtd-theme", "autodocsumm"],
    },
    zip_safe=False,
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    options=options,
)
