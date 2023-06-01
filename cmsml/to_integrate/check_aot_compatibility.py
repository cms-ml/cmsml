from __future__ import annotations  # can be removed for Python 3.10, works only at 3.7+
import tensorflow as tf
from pathlib import Path
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"


class OpsTable():
    supported_devices = ("cpu", "gpu")

    """
    AOT needs two requirements to work:
        1) the outcome of an ops-kernel needs to be deterministic
        2) the ops-kernel needs to have an XLA implementation.

    Tensorflow can return a markdown table containing all XLA compatible ops.
    This class is a wrapper to create this table and consequently read it.
    """

    def __init__(self, *tables: tuple[str]):
        """
        Args:
            tables (tuple[str] | OpsTable): tuple of location of multiple tables or OpsTables
            create_tables (tuple[str], optional): Description

        Deleted Parameters:
            table_dir (str): Path to the markdown table
        """
        super().__init__()
        self.table_locations = tables
        self.ops_dict = {}
        self.merge_markdown_tables()

    @staticmethod
    def _supported_devices(device):
        if device not in OpsTable.supported_devices and not None:
            msg = f"Device {device} is not supported. Supported are: {OpsTable.supported_devices}"
            raise NotImplementedError(msg)

    def _get_unique_ops(self, device: str = None) -> set:
        """
        Get all unique ops from a given *ops_dict*. If *device*
        is used the result is filtered after a specific device.

        Args:
            device (str, optional): Name of a supported device (cpu, gpu)

        Returns:
            set: All unique operators

        Raises:
            NotImplementedError: Raises, when an unsupported device is used.
        """
        OpsTable._supported_devices(device)

        if device is None:
            unique_ops = {value["op"] for value in self.ops_dict.values()}
        else:
            unique_ops = {value["op"] for value in self.ops_dict.values() if value[device]}
        return unique_ops

    @property
    def cpu_ops(self) -> set[str]:
        # get unique XLA compatible results for CPU only
        return self._get_unique_ops("cpu")

    @property
    def gpu_ops(self) -> set[str]:
        # # get unique XLA compatible results for GPU only
        return self._get_unique_ops("gpu")

    @property
    def ops(self) -> set[str]:
        # get unique ops that has CPU and GPU implementation
        return self.cpu_ops.intersection(self.gpu_ops)

    def __len__(self) -> int:
        # number of ops
        return len(self.ops_dict.keys())

    @staticmethod
    def create_markdown_table(device: str = "CPU", save_location=None, suffix: str = ".md"):
        """
        Creates table stream containing all XLA supported ops for given *device*.
        You need to be in the CMSSW software enviroment to run this command.
        The table is saved at *dst*/<current_tensorflow_version>.*suffix*

        Args:
            device (str, optional): CPU or GPU (default: CPU)
            save_location (str): destination of resulting file
            suffix (str, optional): any suffix (default: .md)

        Returns: stream (str): String of markdown table
        """
        import subprocess
        mapping = {"CPU": "XLA_CPU_JIT",
                   "GPU": "XLA_GPU_JIT",
                   }

        # check if enviroment is sourced correctly
        if subprocess.getstatusoutput("tf2xla_supported_ops")[0] == 127:
            msg = "Command tf2xla_supported_ops not found."\
                "Are you sure you sourced your CMSSW enviroment with 'csmenv'?"
            raise OSError(msg)

        # tf2xla_supported_ops prints the table
        # catch the stdout put stream and decode into str
        cmd = ["tf2xla_supported_ops", f"--device={mapping[device]}"]
        out = subprocess.run(cmd, stdout=subprocess.PIPE)
        table = out.stdout.read().decode("utf-8")

        # save table at *save_location*
        try:
            suffix = suffix if suffix.startswith(".") else f".{suffix}"
            dst = Path(save_location).with_stem(
                f"tensorflow_v{tf.__version__}_{mapping[device]}").with_suffix(suffix)
            with open(str(dst), "w") as file:
                file.write(table)
        except FileNotFoundError as error:
            raise(error)

        return table

    @staticmethod
    def read_markdown_table(table: str = None, *, create_table_for_device=None) -> dict:
        """
        Read a *markdown_table* created with 'tf2xla_supported_ops' and returns a dictionary.

        Args:
            table (str): location of the markdown file

        Returns:
            dict: dictionary of ops with allowed types
        """
        def set_device_flags(line: str) -> tuple[str, str]:
            # set cpu and gpu flag depending on first line in markdown
            cpu = "XLA_CPU_JIT" in line
            gpu = "XLA_GPU_JIT" in line
            return cpu, gpu

        # creation of table has higher priority than reading
        if create_table_for_device:
            OpsTable._supported_devices(create_table_for_device)
            lines = OpsTable.create_markdown_table(device=create_table_for_device)
        else:
            with open(table) as file:
                lines = file.read().splitlines(True)

        # filter out ops
        all_ops = lines[4:-6]
        # first line contains device information
        is_cpu, is_gpu = set_device_flags(lines[0])

        ops = {}
        for line_op in all_ops:
            # remove all non-ops chars
            for remove_char in ("`", "\n"):
                line_op = line_op.replace(remove_char, "")
            operation, allowed_types = line_op.split("|")

            # clean up whitespace
            operation = operation.lstrip().rstrip()
            allowed_types = allowed_types.lstrip().rstrip()

            # sets the typed of cpu or gpu
            # its possible to have different allowed types
            op = {"op": operation,
                  "cpu": None,
                  "gpu": None,
                  }

            if is_cpu:
                op["cpu"] = allowed_types
            elif is_gpu:
                op["gpu"] = allowed_types

            ops[operation] = op
        return ops

    def merge_markdown_tables(self) -> dict:
        """
        Merges multiple tables of different devices, but same TF version into 1 dictionary.

        WARNING: Since its not possible to see from which version the markdown table is generate, try
        to not mix tables from differnt tensorflow versions.

        Returns:
            dict: dictionary of ops with allowed types
        """

        def merge_nested_dict(a, b):
            # update dict a inplace with entries in b
            # a and b are nested on first level

            a_keys = a.keys()
            b_keys = b.keys()
            intersection_keys = set(a_keys) & set(b_keys)
            in_b_but_not_a_keys = set(b_keys) - set(a_keys)

            # update ops in a with information from b
            for key in intersection_keys:
                a_inner_dict = a.get(key, None)
                b_inner_dict = b.get(key, None)
                merged_inner_keys = set(a_inner_dict.keys()).union(set(b_inner_dict.keys()))

                for inner_key in merged_inner_keys:
                    a[key][inner_key] = (a_inner_dict.get(inner_key) or b_inner_dict.get(inner_key))

            # add ops to a that are in b but not in a
            for key in in_b_but_not_a_keys:
                a[key] = b[key]

            # read multiple tables

        ops_dicts = [self.read_markdown_table(table) for table in self.table_locations]
        # if not table paths are provide, create the tables for all supported_devices
        if not ops_dicts:
            ops_dicts = [self.read_markdown_table(create_tables=device)
                         for device in self.supported_devices]
        merged_dict = {}
        for ops_dict in ops_dicts:
            merge_nested_dict(merged_dict, ops_dict)
        self.ops_dict.update(merged_dict)

    @staticmethod
    def compare_2_tables(OpsTable1, OpsTable2, verbose=False):
        """
        Print the intersection and mutuale exclusive of two markdown tables.
        This method is nice to see if a new tensorflow version comes
        with new supported ops compared to an older version

        Args:
            first_table (str): Description
            second_table (str): Description

        Deleted Parameters:
            other_table (OpsTable): Another instance with readed OpsTable
        """
        first_table = OpsTable1.ops_dict
        second_table = OpsTable2.ops_dict

        first_keys = first_table.keys()
        second_keys = second_table.keys()
        # exclusive or
        or_keys = first_keys ^ second_keys

        # get intersection, left, right exclusive
        same_ops = sorted(list(first_keys & second_keys))
        own_exclusive_ops = sorted(list(first_keys & or_keys))
        other_exclusive_ops = sorted(list(second_keys & or_keys))

        # print everything in a aligned manner
        if verbose:
            pack = [("Intersection of Ops", same_ops),
                ("Exclusive to Self", own_exclusive_ops),
                ("Exclusive to Other", other_exclusive_ops)]

            for header, ops in pack:
                print(header.center(40, "*"))
                for op in ops:
                    print(op.center(40))
                print("".center(40, "-"))
        return same_ops, own_exclusive_ops, other_exclusive_ops


class ModelOpsChecker():
    # ops are not use by XLA, just placeholder for edges of graph
    placeholder_ops = ("NoOp", "Placeholder")

    def __init__(self,
                model_path: str,
                serving_key: str = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
                node_def_number=0):
        """
        This class taks as input a SavedModel or pb file and returns all unique operators.
        These unique operators are then matched against their XLA compatiblity.

        !!! Be aware, that the compatible list of Ops changes with each tensorflow version. !!!

        Args:
            tf_model (str): path to tf.saved_model
        """
        # None if graph is loaded not Savedmodel
        self.model_path = model_path
        self.model_type = None
        self.serving_key = serving_key
        self.loaded_graph_def = self.load_graph_def(self.model_path, self.serving_key)
        self.node_def_number = node_def_number

    def load_graph_def(self,
                       model_path,
                       serving_key=tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY):
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
        model_path = Path(model_path)
        # if model_path is directory try load as saved model
        if model_path.is_dir() and tf.saved_model.contains_saved_model(str(model_path)):
            # if keras model try to load as keras model
            if (model_path / "keras_metadata.pb").exists():
                loaded_saved_model = tf.keras.models.load_model(str(model_path))
            else:
                loaded_saved_model = tf.saved_model.load(str(model_path))

            # extract graph
            if serving_key not in loaded_saved_model.signatures:
                error_msg = f"Given serving_key: *{serving_key}* is not in loaded model. "\
                    f"Signatures of the loaded model are: {list(loaded_saved_model.signatures)}"
                raise KeyError(error_msg)

            graph_def = loaded_saved_model.signatures[serving_key].graph.as_graph_def()
            self.model_type = 'saved_model'
        # else load as graph
        elif model_path.suffix == '.pb':

            # for frozen graphs only
            with tf.io.gfile.GFile(str(model_path), "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                self.model_type = 'graph'
        else:
            # raise error if neither a frozehn graph or savedmodel is found
            error_msg = f"Given path: {model_path} is neither frozen graph, nor SavedModel."
            raise FileNotFoundError(error_msg)
        return graph_def

    def _get_unique_attribute(self, graph_def, attribute: str, *, node_def_number: int = 0) -> set[str, str]:
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
        node_def = None
        # extract tf.compat.v1.NodeDef from GraphDef.
        if self.model_type == 'saved_model':
            try:
                node_def = graph_def.library.function[node_def_number].node_def
            except:
                # raise error when wrong function library is taken.
                num_funcs = len(graph_def.library.function[node_def_number])
                if num_funcs > 1:
                    msg = f"You have {num_funcs} FunctionDef's."\
                        f" Change your node_def_number to another between 0 and {num_funcs}"
                    raise AttributeError(msg)
            unique_attributs = {getattr(node, attribute) for node in node_def}
        else:
            # frozen graphs have no proper saving of their ops in a node_def
            # ops and names need to be extracted from protobuffer
            unique_attributs = {getattr(node, attribute) for node in graph_def.node}
        return unique_attributs

    @property
    def unique_names(self) -> set[str, str]:
        """
        Return set of all names of all used ops within loaded Model
        """
        return self._get_unique_attribute(self.loaded_graph_def, "name", node_def_number=self.node_def_number)

    @property
    def unique_ops(self) -> set[str, str]:
        """
        Return set of all ops used within loaded Model
        """
        return self._get_unique_attribute(self.loaded_graph_def, "op", node_def_number=self.node_def_number)

    def find_match(self, table_paths: str = None,
                device: str = "cpu") -> list[str, bool]:
        ops_table = OpsTable(table_paths)

        # get uniqure ops from table
        if device.lower() == "cpu":
            unique_table_ops = ops_table.cpu_ops
        elif device.lower() == "gpu":
            unique_table_ops = ops_table.gpu_ops
        else:
            msg = f"Your device {device} is not supported."\
                f"Current supported devices are: {ops_table.supported_devices}"
            raise ValueError(msg)

        # check if graph ops match with table ops
        # result is list with tuples(ops_name, bool)
        graph_ops = self.unique_ops
        compatible_ops_collection = {(graph_op, graph_op in unique_table_ops)
                               for graph_op in graph_ops
                               if graph_op not in self.placeholder_ops}

        return compatible_ops_collection

    def print_compatible_ops(self, table_path: str, device, print_uncompatible_ops: bool = False):
        """

        """

        compatible_set = self.find_match(table_path, device)
        header = ("Operation", "has XLA")

        print(f"{device.upper().center(40)}")
        print(f"{header[0]:20}|{header[1]:5}")
        print("-" * 20 + "|" + "-" * 20)
        for op, status in compatible_set:
            print(f"{op:20}|{str(bool(status)):5}")

        # check if any are not compatible
        if print_uncompatible_ops:
            uncompatible = [op for op, status in compatible_set if not status]
            if uncompatible:
                print("\nFollowing ops are not XLA compatible:", *uncompatible, sep="\n")
            else:
                print("\nAll ops are XLA compatible")


if __name__ == "__main__":
    path_table = "tests/compat_ops_tables/2_6_4_XLA_CPU_JIT.txt"
    gpu_path_table = "tests/compat_ops_tables/XLA_GPU_JIT_old.txt"
    cpu_path_table = "tests/compat_ops_tables/XLA_CPU_JIT_old.txt"

    # gpu_table = OpsTable(gpu_path_table)
    # cpu_table = OpsTable(cpu_path_table)
    t = OpsTable(cpu_path_table, gpu_path_table)

    model_saved_path = "tests/test_models/test_saved_model"
    model_graph_path = "tests/test_models/test_freeze_graph.pb"

    tf_check = ModelOpsChecker(model_saved_path)
    graph_check = ModelOpsChecker(model_graph_path)
    from IPython import embed
    embed()
