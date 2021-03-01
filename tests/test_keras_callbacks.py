# coding: utf-8

"""
Keras callback tests.
"""

import cmsml

from . import CMSMLTestCase, require_py3, require_nvml


class KerasCallbacksTestCase(CMSMLTestCase):

    @require_py3
    @require_nvml
    def test_gpu_stats(self):
        gpu_logger = cmsml.keras.callbacks.GPUStatsLogger()
        logs = {}
        gpu_logger._log_usage(logs)
        self.assertTrue("GPU0 util/%" in logs)
        self.assertTrue("GPU0 mem/MiB" in logs)
        self.assertTrue("GPU0 mem/%" in logs)
        print("\n".join("{} -> {}".format(*tpl) for tpl in logs.items()))

        gpu_logger = cmsml.keras.callbacks.GPUStatsLogger(stats=["mem"])
        logs = {}
        gpu_logger._log_usage(logs)
        self.assertTrue("GPU0 util/%" not in logs)
        self.assertTrue("GPU0 mem/MiB" in logs)
        self.assertTrue("GPU0 mem/%" not in logs)
