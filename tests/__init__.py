# coding: utf-8

"""
Tests.
"""

import os
import sys
import unittest

thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(thisdir))
import cmsml  # noqa


class TestCase(unittest.TestCase):

    def test_test(self):
        pass
