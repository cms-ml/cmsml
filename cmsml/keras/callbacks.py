# coding: utf-8

"""
Custom keras callbacks.
"""

__all__ = ["GPUStatsLogger"]


from cmsml.util import make_list, verbose_import

import tensorflow as tf


class GPUStatsLogger(tf.keras.callbacks.Callback):

    def __init__(self, *args, **kwargs):
        device_numbers = make_list(kwargs.pop("device_numbers", [0]))

        super(GPUStatsLogger, self).__init__(*args, **kwargs)

        # setup py3nvml
        py3nvml = verbose_import("py3nvml", pip_name="py3nvml", user=self.__class__.__name__)
        self.smi = py3nvml.py3nvml
        self.smi.nvmlInit()

        # get device handles
        self.device_numbers = device_numbers
        self.handles = [self.smi.nvmlDeviceGetHandleByIndex(n) for n in self.device_numbers]

    def _log_usage(self, logs):
        for n, handle in zip(self.device_numbers, self.handles):
            mem = self.smi.nvmlDeviceGetMemoryInfo(handle)
            res = self.smi.nvmlDeviceGetUtilizationRates(handle)
            logs["GPU {} usage [%]".format(n)] = res.gpu
            logs["GPU {} vRAM [%]".format(n)] = 100 * mem.used / mem.total
            logs["GPU {} vRAM [MB]".format(n)] = mem.used / (1024**2)

    def on_batch_end(self, x, logs=None):
        if logs is not None:
            self._log_usage(logs)
