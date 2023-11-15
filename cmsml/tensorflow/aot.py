# coding: utf-8

"""
Tools and objects for working with AOT / XLA.
"""

from __future__ import annotations

import sys
import re
from subprocess import PIPE

from cmsml.util import interruptable_popen
from cmsml.tensorflow.tools import import_tf

tf = import_tf()[0]

from tensorflow.core.framework.graph_pb2 import GraphDef


class OpsData(object):
    """
    AOT needs two requirements to work:
        1) the outcome of an ops-kernel needs to be deterministic
        2) the ops-kernel needs to have an XLA implementation.

    Tensorflow can return a markdown table containing all XLA compatible ops.
    This class is a wrapper to create this table and consequently read it.
    """

    device_ids = {
        "cpu": "XLA_CPU_JIT",
        "gpu": "XLA_GPU_JIT",
    }

    def __init__(self: OpsData, devices: tuple[str] | None = None) -> None:
        """
        Sets an iterable of *devices* for which the XLA operations table should be generate.
        """
        super().__init__()

        # store operation data in a nested dict
        self._ops = {}

        # determine ops
        if not devices:
            devices = ()
        elif not isinstance(devices, (list, tuple, set)):
            devices = (devices,)
        self._determine_ops(devices)

    @classmethod
    def _assert_device_supported(cls, device: str) -> None:
        if device not in cls.device_ids:
            raise ValueError(
                f"{device} not in supported devices {list(cls.device_ids.keys())}",
            )

    @classmethod
    def read_ops_table(
        cls,
        device: str = "cpu",
    ) -> str:
        """
        Generate a markdown table for *device* and returns it.
        """
        cls._assert_device_supported(device)

        # tf2xla_supported_ops prints the table
        # catch the stdout put stream and decode into str
        cmd = f"tf2xla_supported_ops --device={cls.device_ids[device]}"
        code, out, _ = interruptable_popen(cmd, stdout=PIPE, executable="/bin/bash", shell=True)
        if code != 0:
            raise Exception(f"tf2xla_supported_ops command failed with exit code {code}")

        return out

    @classmethod
    def parse_ops_table(
        cls,
        table: str | None = None,
        *,
        device: str = "cpu",
    ) -> dict[str, dict]:
        """
        Read a given markdown-*table* generated with 'tf2xla_supported_ops' and returns a dictionary contaning all ops
        with XLA implementation. For a given table the *device* information is ignored and extracted from the table. If
        no table is given one will be generate for given *device*.
        """
        cls._assert_device_supported(device)

        # create the table if empty
        if not table:
            table = cls.read_ops_table(device)
        else:
            with open(table, "r") as txt_file:
                table = txt_file.read()

        # split into lines
        lines = table.splitlines()

        # first line contains device information
        for device, device_id in cls.device_ids.items():
            if device_id in lines[0]:
                break
        else:
            raise ValueError(f"no device string found in table header '{lines[0]}'")

        # read op infos from table lines
        ops = {}
        content_started = False
        cre = re.compile(r"^\`([^\`]+)\`\s+\|\s*(.*)$")
        for line in lines[1:]:
            line = line.strip()

            # find the beginning of the table
            if not content_started:
                if line.startswith("---"):
                    content_started = True
                continue

            # check if the end is reached
            if not line:
                break

            # parse the line
            m = cre.match(line)
            if not m:
                print(f"error parsing table line: {line}", file=sys.stderr)
                continue

            op_name, allowed_types = m.groups()
            allowed_types = allowed_types.replace("`", "").replace("<br>", "")

            # save op data
            ops[op_name] = {
                "name": op_name,
                "device": device,
                "allowed_types": allowed_types,
            }

        return ops

    def _determine_ops(self: OpsData, devices: tuple[str] | None = None) -> None:
        """
        Merges multiple tables of different devices into 1 dictionary.

        WARNING: Since its not possible to see from which version the markdown table is generated, try to not mix tables
        from different tensorflow versions.
        """
        if not devices:
            devices = tuple(self.device_ids.keys())

        # read op dictionaries
        all_op_dicts = [
            self.parse_ops_table(device=device)
            for device in devices
        ]

        # merge
        ops = {}
        for op_dicts in all_op_dicts:
            for op_data in op_dicts.values():
                op_name = op_data["name"]
                if op_name not in ops:
                    ops[op_name] = {}
                ops[op_name][op_data["device"]] = op_data["allowed_types"]

        self._ops = ops

    def _get_unique_ops(self: OpsData, device: str | None = None) -> set[str]:
        self._assert_device_supported(device)

        return {
            op_name
            for op_name, op_data in self._ops.items()
            if device is None or op_data.get(device)
        }

    @property
    def cpu_ops(self: OpsData) -> set[str]:
        # get unique XLA compatible results for CPU only
        return self._get_unique_ops("cpu")

    @property
    def gpu_ops(self: OpsData) -> set[str]:
        # get unique XLA compatible results for GPU only
        return self._get_unique_ops("gpu")

    @property
    def ops(self: OpsData) -> set[str]:
        # get unique ops that have CPU or GPU implementation
        return self._ops

    def __len__(self: OpsData) -> int:
        # number of ops
        return len(self._ops)

    def __getitem__(self: OpsData, key: str) -> dict:
        return self._ops[key]

    def keys(self: OpsData) -> list[str]:
        return list(self._ops.keys())

    def values(self: OpsData) -> list[dict]:
        return list(self._ops.values())

    def items(self: OpsData) -> list[tuple[str, dict]]:
        return list(self._ops.items())

    def get(self: OpsData, *args, **kwargs) -> tuple[str, dict]:
        return self._ops.get(*args, **kwargs)


def get_graph_ops(graph_def: GraphDef, node_def_number: int = 0) -> list[str]:
    """
    Extracts all ops from a *graph_def* and returns them as a list.
    If there are multiple ``FunctionDef`` instances in the graph, set *node_def_number* to specify from which GraphDef
    the ops should be extracted.
    """
    # extract node definition from the graph "library for savedmodels"
    num_funcs = len(graph_def.library.function)
    # library is empty for graph.pb, but not for SavedModels
    if num_funcs == 0:
        node_def = graph_def.node
    else:
        if node_def_number + 1 > num_funcs:
            raise AttributeError(
                f"node_def_number {node_def_number} does not match amount of {num_funcs} "
                "FunctionDef objects in graph",
            )
        node_def = graph_def.library.function[node_def_number].node_def

    op_names = [node.op for node in node_def]

    return sorted(set(op_names), key=op_names.index)
