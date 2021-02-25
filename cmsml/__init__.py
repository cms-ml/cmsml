# coding: utf-8
# flake8: noqa

"""
Main package file containing provisioning imports.
"""

__all__ = ["__version__"]


# provisioning imports
from cmsml.__meta__ import __version__
import cmsml.util

# start the lazy importer to keep the global the global scope clean and prevent
# imports of packages that might not exist for all users
from cmsml import lazy_importer
lazy_importer.start(["tensorflow", "keras"])
