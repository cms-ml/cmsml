# coding: utf-8

"""
Lazy loader tests.
"""

import cmsml

from . import CMSMLTestCase


class LazyLoaderTestCase(CMSMLTestCase):

    def test_started(self):
        self.assertTrue(cmsml.lazy_loader.started())

    def test_modules(self):
        # check if placeholders are in place
        for module_name in cmsml.lazy_loader._lazy_modules:
            self.assertIsInstance(getattr(cmsml, module_name), cmsml.lazy_loader.ModulePlaceholder)

        # access modules
        for module_name in cmsml.lazy_loader._lazy_modules:
            module = getattr(cmsml, module_name)
            self.assertIsInstance(module.__all__, list)

        # after the first access, the placeholders should have been replaced
        for module_name in cmsml.lazy_loader._lazy_modules:
            module = getattr(cmsml, module_name)
            self.assertNotIsInstance(module, cmsml.lazy_loader.ModulePlaceholder)
