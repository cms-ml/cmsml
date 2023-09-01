# coding: utf-8

from __future__ import annotations

import os
import subprocess

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

import tensorflow as tf
import tabulate


class OpsTable(object):
    """
    AOT needs two requirements to work:
        1) the outcome of an ops-kernel needs to be deterministic
        2) the ops-kernel needs to have an XLA implementation.

    Tensorflow can return a markdown table containing all XLA compatible ops.
    This class is a wrapper to create this table and consequently read it.
    """

    HEADER_LINES = 4
    FOOTER_LINES = 6

    device_mapping = {"cpu": "XLA_CPU_JIT", "gpu": "XLA_GPU_JIT", "tpu": "XLA_TPU_JIT"}

    def __init__(self, devices: list[str] | None = None):
        """
        Args:
            paths (tuple[str] | OpsTable): tuple of locations of multiple tables or OpsTables
            create_tables (tuple[str], optional): Description

        Deleted Parameters:
            table_dir (str): Path to the markdown table
        """
        super().__init__()

        self._ops = {}

        devices = devices or []
        if not isinstance(devices, (list, tuple, set)):
            devices = [devices]
        self._merge_ops(devices)

    @classmethod
    def _assert_device_supported(cls, device: str) -> None:
        if device not in cls.device_mapping:
            raise ValueError(
                f"{device} not in supported devices {list(cls.device_mapping.keys())}",
            )

    @classmethod
    def create_markdown_table(
        cls,
        device: str = "cpu",
    ) -> str:
        """
        Creates table stream containing all XLA supported ops for given *device*.
        You need to be in the CMSSW software enviroment to run this command.
        The table is saved at *dst*/<current_tensorflow_version>.*suffix*

        Args:
            device (str, optional): cpu or gpu (default: cpu)
            path (str, optional): destination of resulting file
            suffix (str, optional): any suffix (default: .md)

        Returns: stream (str): String of markdown table
        """
        cls._assert_device_supported(device)

        # tf2xla_supported_ops prints the table
        # catch the stdout put stream and decode into str
        cmd = f"tf2xla_supported_ops --device={cls.device_mapping[device]}"
        p = subprocess.run(cmd, stdout=subprocess.PIPE, executable="/bin/bash", shell=True)
        if p.returncode != 0:
            raise Exception(f"tf2xla_supported_ops command failed with exit code {p.returncode}")
        table = p.stdout.decode("utf-8")

        return table

    @classmethod
    def parse_markdown_table(
        cls,
        table: str | None = None,
        *,
        device: str = "cpu",
    ) -> dict[str, dict]:
        """
        Read a *markdown_table* created with 'tf2xla_supported_ops' and returns a dictionary.

        Args:
            path (str): location of the markdown file

        Returns:
            dict: dictionary of ops with allowed types
        """
        cls._assert_device_supported(device)

        if not table:
            table = cls.create_markdown_table(device)

        # split into lines
        lines = table.splitlines()

        # first line contains device information
        if "XLA_CPU_JIT" in lines[0]:
            device = "cpu"
        elif "XLA_GPU_JIT" in lines[0]:
            device = "gpu"
        elif "XLA_TPU_JIT" in lines[0]:
            device = "tpu"
        else:
            raise ValueError(f"no device string found in table header '{lines[0]}'")

        ops = {}
        for line_op in lines[cls.HEADER_LINES:-cls.FOOTER_LINES]:
            # remove all non-ops chars
            for remove_char in ("`",):
                line_op = line_op.replace(remove_char, "")
            operation, allowed_types = line_op.split("|")

            # clean up whitespace
            operation = operation.lstrip().rstrip()
            allowed_types = allowed_types.lstrip().rstrip()

            # sets the typed of cpu or gpu
            # its possible to have different allowed types
            op = {
                "name": operation,
                "device": device,
                "allowed_types": allowed_types,
            }

            ops[operation] = op

        return ops

    def _merge_ops(self, devices: list[str]) -> None:
        """
        Merges multiple tables of different devices into 1 dictionary.

        WARNING: Since its not possible to see from which version the markdown table is generated,
        try to not mix tables from differnt tensorflow versions.
        """
        if not devices:
            devices = list(self.device_mapping.keys())

        # read op dictionaries
        all_op_dicts = [
            self.parse_markdown_table(device=device)
            for device in devices
        ]

        # merge
        merged = {}
        for op_dicts in all_op_dicts:
            for op_data in op_dicts.values():
                op_name = op_data["name"]
                if op_name not in merged:
                    merged[op_name] = {}
                merged[op_name][op_data["device"]] = op_data["allowed_types"]

        self._ops = merged

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
            value["op"]
            for value in self._ops.values()
            if device is None or value[device]
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
    def tpu_ops(self) -> set[str]:
        # get unique XLA compatible results for TPU only
        return self._get_unique_ops("tpu")

    @property
    def ops(self) -> set[str]:
        # get unique ops that have CPU, GPU or TPU implementation
        return self._get_unique_ops()

    def __len__(self) -> int:
        # number of ops
        return len(self._ops)

    def __getitem__(self, key: str) -> dict:
        return self._ops[key]


class ModelOpsChecker(object):

    # ops that are not used by XLA, just placeholders for edges of graph
    placeholder_ops = ("NoOp", "Placeholder")

    def __init__(
        self,
        model_path: str,
        serving_key: str = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
    ):
        """
        This class taks as input a SavedModel or pb file and returns all unique operators.
        These unique operators are then matched against their XLA compatiblity.

        !!! Be aware, that the compatible list of Ops changes with each tensorflow version. !!!

        Args:
            tf_model (str): path to tf.saved_model
        """
        super().__init__()

        self.model_type = None
        self.graph_def = None

        self._load_graph_def(model_path, serving_key)

    def _load_graph_def(
        self,
        model_path: str,
        serving_key: str = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
    ):
        """Load model from *model_path* and extracts the graph_def.
        If the model is a SavedModel, *serving_key* needs to pass along.

        Args:
            model_path (str): Path to SavedModel directory or graph.pb file
            serving_key (str, Optional): Serving key under which a concrete function is saved, only
            necessary for SavedModels.

        Returns:
            TYPE: GraphDef

        Raises:
            FileNotFoundError: Raised if *model_path* is neither SavedModel nor Graph
            KeyError: Raised if *serving_key* does not exist in given model
        """
        model_path = os.path.expandvars(os.path.expanduser(model_path))

        # if model_path is directory try load as saved model
        if model_path.is_dir() and tf.saved_model.contains_saved_model(model_path):
            # if keras model try to load as keras model
            if os.path.exists(os.path.join(model_path, "keras_metadata.pb")):
                loaded_saved_model = tf.keras.models.load_model(model_path)
            else:
                loaded_saved_model = tf.saved_model.load(model_path)

            # extract graph
            if serving_key not in loaded_saved_model.signatures:
                raise KeyError(
                    f"Given serving_key: *{serving_key}* is not in loaded model. "
                    f"Signatures of the loaded model are: {list(loaded_saved_model.signatures)}",
                )

            self.model_type = "saved_model"
            self.graph_def = loaded_saved_model.signatures[serving_key].graph.as_graph_def()
            return

        # load as graph
        if os.path.splitext(model_path)[1] == ".pb":  # pb.txt pbtxt??
            with tf.io.gfile.GFile(str(model_path), "rb") as f:
                self.graph_def = tf.compat.v1.GraphDef()
                self.graph_def.ParseFromString(f.read())

            self.model_type = "graph"
            return

        # raise error if neither a frozehn graph or savedmodel is found
        raise FileNotFoundError(f"{model_path} is neither frozen graph nor SavedModel")

    def _get_graph_attributes(self, attribute: str, *, node_def_number: int = 0) -> set[str]:
        """
        Get the *attribute* of all Nodes in *graph_def*. Valid attributes are for example: 'name' or
        'ops'. The node_def_number is only necessary for Graph_defs coming from SavedModels.
        For graph_defs coming from frozen graphs *node_def_number* is not used.
        name: Unique string identifier for the node.
        op: String that specifies the type of operation.
        input: List of strings that specify the input tensors for the node.
        device: Optional string that specifies the device on which the operation should be executed
        attr: Map of attribute names to attribute values. Provide additional information about the op

        Args:
            attribute (str): Name of the attribute within NodeDef to look for (typical: ops or names)
            node_def_number (int, optional): Number of the function def to look up (default: 0)

        Returns:
            set[str, str]: Description

        Raises:
            AttributeError: Raises Error if node_def_number exceeds the number of FunctionDefs.
        """
        # get tf.compat.v1.GraphDef
        node_def = self.graph_def.node

        # extract tf.compat.v1.NodeDef from GraphDef.
        if self.model_type == "saved_model":
            num_funcs = len(self.graph_def.library.function)
            if node_def_number + 1 > num_funcs:
                raise AttributeError(
                    f"You have {num_funcs} FunctionDef's. "
                    f"Change your node_def_number to another between 0 and {num_funcs - 1}",
                )
            node_def = self.graph_def.library.function[node_def_number].node_def

        return {getattr(node, attribute) for node in node_def}

    def get_unique_names(self, node_def_number: int = 0) -> set[str]:
        """
        Return set of all names of all used ops within loaded Model
        """
        return self._get_graph_attributes("name", node_def_number=node_def_number)

    def get_unique_ops(self, node_def_number: int = 0) -> set[str]:
        """
        Return set of all ops used within loaded Model
        """
        return self._get_graph_attributes("op", node_def_number=node_def_number)

    def find_match(self, device: str = "cpu") -> set[str, bool]:
        ops_table = OpsTable(device)

        # get uniqure ops from table
        if device.lower() == "cpu":
            unique_table_ops = ops_table.cpu_ops
        elif device.lower() == "gpu":
            unique_table_ops = ops_table.gpu_ops
        else:  # tpu
            unique_table_ops = ops_table.tpu_ops

        # check if graph ops match with table ops
        # result is list with tuples(ops_name, bool)
        compatible_ops_collection = {
            (graph_op, graph_op in unique_table_ops)
            for graph_op in self.get_unique_ops()
            if graph_op not in self.placeholder_ops
        }

        return compatible_ops_collection

    def print_compatible_ops(self, device: str = "cpu"):
        """
        TODO.
        """
        header = ("Operation", f"XLA/AOT supported on {device}")
        content = []
        incompatible_ops = []
        for op, status in sorted(self.find_match(device)):
            content.append((op, status))
            if not status:
                incompatible_ops.append(op)

        print(tabulate.tabulate(content, headers=header))

        # check if any are not compatible
        if incompatible_ops:
            print("\nFollowing ops are not XLA compatible:", *incompatible_ops, sep="\n")
        else:
            print("\nAll ops are XLA compatible")


if __name__ == "__main__":
    path_table = "tests/compat_ops_tables/2_6_4_XLA_CPU_JIT.txt"
    gpu_path_table = "tests/compat_ops_tables/XLA_GPU_JIT_old.txt"
    cpu_path_table = "tests/compat_ops_tables/XLA_CPU_JIT_old.txt"

    # gpu_table = OpsTable(gpu_path_table)
    # cpu_table = OpsTable(cpu_path_table)
    t = OpsTable()
    from IPython import embed; embed()

    # model_saved_path = "tests/test_models/test_saved_model"
    # model_graph_path = "tests/test_models/test_freeze_graph.pb"

    # tf_check = ModelOpsChecker(model_saved_path)
    # graph_check = ModelOpsChecker(model_graph_path)
    # from IPython import embed
    # embed()
