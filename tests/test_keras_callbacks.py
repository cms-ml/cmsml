# coding: utf-8

"""
TensorFlow tests.
"""

import os

import cmsml

from . import CMSMLTestCase, require_py3, require_nvml


class KerasCallbacksTestCase(CMSMLTestCase):

    @require_py3
    @require_nvml
    def test_todo(self):
        gpu_logger = cmsml.keras.callbacks.GPUStatsLogger(device_numbers=[0])

        logs = {}
        gpu_logger._log_usage(logs)
        self.assertTrue("GPU 0 usage [%]" in logs)
        self.assertTrue("GPU 0 vRAM [%]" in logs)
        self.assertTrue("GPU 0 vRAM [MB]" in logs)
