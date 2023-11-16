# coding: utf-8

"""
Lazy loader tests.
"""

import cmsml

from . import CMSMLTestCase


class LazyLoaderTestCase(CMSMLTestCase):

    def test_started(self):
        self.assertTrue(cmsml.lazy_loader.started())
