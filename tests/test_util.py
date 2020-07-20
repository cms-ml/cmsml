# coding: utf-8

"""
Util tests.
"""

import os
import shutil
import unittest

import cmsml


class UtilTestCase(unittest.TestCase):

    def test_tmp_file(self):
        with cmsml.util.tmp_file(create=False, delete=False) as path:
            self.assertFalse(os.path.exists(path))

        with cmsml.util.tmp_file(create=True, delete=True) as path:
            self.assertTrue(os.path.exists(path))
        self.assertFalse(os.path.exists(path))

        with cmsml.util.tmp_file(create=True, delete=False) as path:
            self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.exists(path))
        os.remove(path)
        self.assertFalse(os.path.exists(path))

        with cmsml.util.tmp_file(create=False, delete=True) as path:
            self.assertFalse(os.path.exists(path))
        self.assertFalse(os.path.exists(path))

    def test_tmp_dir(self):
        with cmsml.util.tmp_dir(create=False, delete=False) as path:
            self.assertFalse(os.path.exists(path))

        with cmsml.util.tmp_dir(create=True, delete=True) as path:
            self.assertTrue(os.path.exists(path))
        self.assertFalse(os.path.exists(path))

        with cmsml.util.tmp_dir(create=True, delete=False) as path:
            self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.exists(path))
        shutil.rmtree(path)
        self.assertFalse(os.path.exists(path))

        with cmsml.util.tmp_dir(create=False, delete=True) as path:
            self.assertFalse(os.path.exists(path))
        self.assertFalse(os.path.exists(path))
