# coding: utf-8

"""
Custom keras callbacks.
"""

__all__ = []


from cmsml.util import make_list, verbose_import

import tensorflow as tf


class GPUStatsLogger(tf.keras.callbacks.Callback):
    """ __init__(device_numbers=[0], stats=["util", "mem", "mem_rel"], *args, **kwargs)
    Keras callback that, per used GPU, logs its usage statistics after each batch. The list of used
    GPUs can be configured with *device_numbers* and should be a list of integers. The fields to log
    are selectable via *stats*. Available fields are:

    - ``util``: The volatile GPU utilization in per cent.
    - ``mem``: The amount of memory currently blocked on the GPU in MiB.
    - ``mem_rel``: The amount of memory currently blocked on the GPU in per cent, relative to the
      total amount of available memory.

    **Requires the** `py3nvml <https://pypi.org/project/py3nvml/>`__ **package.**

    .. note::
        Please note that the amount of reported memory might also depend on other processes using
        the same GPU and is not necessarily allocated solely to the model that uses this callback.

    Examle usage:

    .. code-block:: python

        # enable the "util" and "mem" logs for GPU device 0 (the default).
        gpu_stats = GPUStatsLogger(stats=["util", "mem"])

        # enable all logs for GPUs 0 and 2
        gpu_stats = GPUStatsLogger(device_numbers=[0, 2])
    """

    AVAILABLE_STATS = ["util", "mem", "mem_rel"]

    def __init__(self, *args, **kwargs):
        # get configs from kwargs
        device_numbers = make_list(kwargs.pop("device_numbers", [0]))
        stats = make_list(kwargs.pop("stats", self.AVAILABLE_STATS))

        # make device_numbers unique
        device_numbers = list(sorted(set(device_numbers), key=lambda n: device_numbers.index(n)))

        # validate stat
        for s in stats:
            if s not in self.AVAILABLE_STATS:
                raise ValueError("unknown GPU stats '{}', valid values are: {}".format(
                    s, ", ".join(self.AVAILABLE_STATS)))

        # super init
        super(GPUStatsLogger, self).__init__(*args, **kwargs)

        # store configs
        self.device_numbers = device_numbers
        self.stats = stats

        # setup py3nvml
        py3nvml = verbose_import("py3nvml", pip_name="py3nvml", user=self.__class__.__name__)
        self.smi = py3nvml.py3nvml
        self.smi.nvmlInit()

        # get device handles
        self.handles = [self.smi.nvmlDeviceGetHandleByIndex(n) for n in self.device_numbers]

    def _log_usage(self, logs):
        # nothing to do when no stats should be shown
        if not self.stats:
            return

        # keep track of key-value-unit triplets of the stats
        data = []

        # add stats per device handle
        for n, handle in zip(self.device_numbers, self.handles):
            # get device stats
            mem = self.smi.nvmlDeviceGetMemoryInfo(handle)
            util = self.smi.nvmlDeviceGetUtilizationRates(handle)

            # add stats to logs in the configured order
            for s in self.stats:
                if s == "util":
                    data.append(("gpu{} util".format(n), util.gpu, "%"))
                elif s == "mem":
                    data.append(("gpu{} mem".format(n), mem.used / 1024**2., "MiB"))
                elif s == "mem_rel":
                    data.append(("gpu{} mem".format(n), 100 * mem.used / mem.total, "%"))

        # add all of them to the logs with units added to keys
        logs.update({"{}/{}".format(k, u): v for k, v, u in data})

        # print the stats line
        stats_line = ", ".join("{}: {:.1f}{}".format(*d) for d in data)
        print("\nGPU stats --- " + stats_line)

    def on_epoch_end(self, i, logs=None):
        """
        Logs the GPU stats after each epoch *i* and stores them in *logs* when set.
        """
        if logs is not None:
            self._log_usage(logs)
