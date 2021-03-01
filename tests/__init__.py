# coding: utf-8
# flake8: noqa

"""
Tests.
"""

import os
import sys

thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(thisdir))

from .test_lazy_loader import *
from .test_util import *
from .test_tensorflow import *
