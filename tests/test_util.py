# coding: utf-8

"""
Util tests.
"""

import os
import shutil

import six

import cmsml

from . import CMSMLTestCase


class UtilTestCase(CMSMLTestCase):

    def test_is_lazy_iterable(self):
        if six.PY3:
            d = {1: 2}
            self.assertTrue(cmsml.util.is_lazy_iterable(d.keys()))
            self.assertTrue(cmsml.util.is_lazy_iterable(d.values()))
            self.assertTrue(cmsml.util.is_lazy_iterable(range(10)))
            self.assertTrue(cmsml.util.is_lazy_iterable(enumerate(range(10))))
        else:
            self.assertTrue(cmsml.util.is_lazy_iterable(six.moves.range(10)))
        self.assertTrue(i for i in range(10))

    def test_make_list(self):
        self.assertIsInstance(cmsml.util.make_list([]), list)
        self.assertIsInstance(cmsml.util.make_list(()), list)
        self.assertIsInstance(cmsml.util.make_list(set()), list)
        self.assertIsInstance(cmsml.util.make_list((i for i in range(10))), list)
        self.assertIsInstance(cmsml.util.make_list(1), list)
        self.assertIsInstance(cmsml.util.make_list("s"), list)

    def test_verbose_import(self):
        cmsml.util.verbose_import("os")

        with self.assertRaises(ImportError):
            cmsml.util.verbose_import("not_existing")

        with self.assertRaises(ImportError) as cm:
            cmsml.util.verbose_import("not_existing", user="USER", pip_name="not_existing")
        self.assertTrue("but is required by USER" in str(cm.exception))
        self.assertTrue("pip install --user not_existing" in str(cm.exception))

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
