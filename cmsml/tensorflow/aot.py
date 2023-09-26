# coding: utf-8

"""
Tools and objects for working with AOT / XLA.
"""

from __future__ import annotations

import os
import sys
import re
from subprocess import PIPE

from cmsml.util import interruptable_popen
from cmsml.tensorflow.tools import import_tf


tf = import_tf()[0]


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

    def __init__(self, devices: tuple[str] | None = None):
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
        Read a given markdown-*table* generated with 'tf2xla_supported_ops' and returns a dictionary
        contaning all ops with XLA implementation.
        For a given table the *device* information is ignored and extracted from the table.
        If not table is given one will be generate for given *device*.
        """
        cls._assert_device_supported(device)

        # create the table if empty
        if not table:
            table = cls.read_ops_table(device)

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

    def _determine_ops(self, devices: tuple[str] | None = None) -> None:
        """
        Merges multiple tables of different devices into 1 dictionary.

        WARNING: Since its not possible to see from which version the markdown table is generated,
        try to not mix tables from differnt tensorflow versions.
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

    def _get_unique_ops(self, device: str | None = None) -> set[str]:
        """
        Get all unique ops. If *device* is used the result is filtered after a specific device.

        Args:
            device (str, optional): Name of a supported device (cpu, gpu)

        Returns:
            set: All unique operators

        Raises:
            NotImplementedError: Raises, when an unsupported device is used.
        """
        self._assert_device_supported(device)

        return {
            op_name
            for op_name, op_data in self._ops.items()
            if device is None or op_data.get(device)
        }

    @property
    def cpu_ops(self) -> set[str]:
        # get unique XLA compatible results for CPU only
        return self._get_unique_ops("cpu")

    @property
    def gpu_ops(self) -> set[str]:
        # get unique XLA compatible results for GPU only
        return self._get_unique_ops("gpu")

    @property
    def ops(self) -> set[str]:
        # get unique ops that have CPU or GPU implementation
        return self._get_unique_ops()

    def __len__(self) -> int:
        # number of ops
        return len(self._ops)

    def __getitem__(self, key: str) -> dict:
        return self._ops[key]

    def keys(self) -> list[str]:
        return self._ops.keys()

    def values(self) -> list[dict]:
        return self._ops.values()

    def items(self) -> list[tuple[str, dict]]:
        return self._ops.items()

    def get(self, *args, **kwargs) -> tuple[str, dict]:
        return self._ops.get(*args, **kwargs)


def load_graph_def(
    model_path: str,
    serving_key: str = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
) -> tf.compat.v1.GraphDef:
    """
    Loads the model under *model_path* and returns the GraphDef of it.
    Support is given for either tensorflow or keras SavedModel, as well as for frozen graphs.

    TODO: merge this with existing function in tools.py?
    """
    model_path = os.path.expandvars(os.path.expanduser(str(model_path)))

    # if model_path is directory try load as saved model
    if os.path.isdir(model_path) and tf.saved_model.contains_saved_model(model_path):
        # if keras model try to load as keras model
        # else load as tensorflow saved model
        loaded_saved_model = load_model(model_path)

        # extract graph
        if serving_key not in loaded_saved_model.signatures:
            raise KeyError(
                f"no graph with serving key '{serving_key}' in model, "
                f"existing keys: {', '.join(list(loaded_saved_model.signatures))}",
            )

        return loaded_saved_model.signatures[serving_key].graph.as_graph_def()

    # load as frozen graph
    if os.path.splitext(model_path)[1] == ".pb":  # pb.txt pbtxt?? TODO
        with tf.io.gfile.GFile(str(model_path), "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        return graph_def

    raise FileNotFoundError(f"{model_path} contains neither frozen graph nor SavedModel")


def load_model(model_path: str) -> tf.Model:
    """
    Load and return the SavedModel stored in the directory *model_path*.
    If the model was saved using Keras API, it will be loaded using the same API, otherwise TensorFlows SavedModel API is used.
    """
    model_path = os.path.expandvars(os.path.expanduser(str(model_path)))

    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "keras_metadata.pb")):
        model = tf.keras.models.load_model(model_path)
    else:
        model = tf.saved_model.load(model_path)

    return model


def get_graph_ops(graph_def: object, node_def_number: int = 0) -> list[str]:
    """
    Reads *graph_def* and extracts all ops from the 'GraphDef'.
    If there are multiple 'FunctionDef' instances in the GraphDef, set *node_def_number* to specify
    from which GraphDef the ops will be extracted.
    """
    node_def = graph_def.node

    # when the node definition is missing, try to extract it from the graph "library"
    if not node_def:
        num_funcs = len(graph_def.library.function)
        if node_def_number + 1 > num_funcs:
            raise AttributeError(
                f"node_def_number {node_def_number} does not match amount of {num_funcs} "
                "FunctionDef objects in graph",
            )
        node_def = graph_def.library.function[node_def_number].node_def

    op_names = [node.op for node in node_def]

    return sorted(set(op_names), key=op_names.index)
