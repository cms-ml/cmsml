# coding: utf-8
# flake8: noqa

"""
Tests.
"""

import os
import sys
import unittest
import functools

import six


class CMSMLTestCase(unittest.TestCase):
    pass


def test_filter(filter_func):
    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            if filter_func(test_func, *args, **kwargs):
                return test_func(*args, **kwargs)
            else:
                print("skipped {}, {}".format(test_func.__name__, filter_func.__name__))
        return wrapper
    return decorator


@test_filter
def require_py3(*args, **kwargs):
    return six.PY3


@test_filter
def require_nvml(*args, **kwargs):
    try:
        from py3nvml import py3nvml
        py3nvml.nvmlInit()
        return True
    except:
        return False


# import all test cases
thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(thisdir))

from .test_lazy_loader import *
from .test_util import *
from .test_tensorflow import *
from .test_keras_callbacks import *
