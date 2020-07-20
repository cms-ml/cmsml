# coding: utf-8

"""
Util tests.
"""

import unittest

import cmsml


class UtilTestCase(unittest.TestCase):

    def test_import_tf(self):
        tf, tf1, tf_version = cmsml.util.import_tf()

        self.assertEqual(len(tf_version), 3)

        if tf_version[0] == "1":
            self.assertEqual(tf, tf1)
