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
        2) the ops-Kernel needs to have an XLA implementation.

    Tensorflow can return a markdown table containing all XLA compatible ops.
    This class is a wrapper to create this table and consequently read it.
    """

    def __init__(self, tables: tuple[str] | OpsTable = None, create_tables: tuple[bool] = None):
        """
        Args:
            tables (tuple[str] | OpsTable): tuple of location of multiple tables or OpsTables
            create_tables (tuple[str], optional): Description

        Deleted Parameters:
            table_dir (str): Path to the markdown table
        """
        super().__init__()
        self.table_locations = [] if tables is None else tables
        self.ops_dict = {}

    def get_unique_ops(self, ops_dict: dict, device: str = None) -> set:
        """
        Get all unique ops from a given *ops_dict*. If *device*
        is used the result is filtered after a specific device.

        Args:
            ops_dict (dict): Dictionary created by reading the markdown table created using bazel
            device (str, optional): Name of a supported device (cpu, gpu)

        Returns:
            set: All unique operators

        Raises:
            NotImplementedError: Raises, when an unsupported device is used.
        """
        if device not in self.supported_devices and not None:
            msg = f"The device {device} is not supported. Supported are: {self.supported_devices}"
            raise NotImplementedError(msg)

        if device is None:
            unique_ops = {value["op"] for value in ops_dict.values()}
        elif ops_dict["op"][device]:
            unique_ops = {value["op"] for value in ops_dict.values() if value[device]}
        return unique_ops

    @property
    def ops(self) -> set[str]:
        return self.get_unique_ops(self.opt_dict)

    @property
    def cpu_ops(self) -> set[str]:
        # get unique XLA compatible results for CPU only
        return self.get_unique_ops(self.opt_dict, "cpu")

    @property
    def gpu_ops(self) -> set[str]:
        # # get unique XLA compatible results for GPU only
        return self.get_unique_ops(self.opt_dict, "gpu")

    @property
    def both_devices_ops(self) -> set[str]:
        # get unique ops that has CPU and GPU implementation
        return self.cpu_ops.intersection(self.gpu_ops)

    def __len__(self) -> int:
        # number of ops
        return len(self.ops_dict.keys())

    def add_table(self, path: str) -> None:
        """
        Add table path to current instance.

        Args:
            path (str): path to table

        Raises:
            FileNotFoundError: Path does not exist
        """
        if Path(path).exists():
            self.table_locations.append(str(path))
        else:
            raise FileNotFoundError(f"path: {path} does not exist.")

    @staticmethod
    def create_markdown_table(dst: str, devices: str = "CPU", suffix: str = ".md"):
        """
        This function creates a markdown table with all XLA supported ops for given *devices*.
        You need to be in the CMSSW software enviroment.
        The table is saved at *dst* with ending *suffix*

        Currently only CPU is supported in CMMSW.
        Args:
            dst (str): destination of resulting file
            device (str, list, tuple): CPU, GPU (default: CPU)
            suffix (str, optional): any suffix (default: .md)
        """
        import subprocess
        mapping = {"CPU": "XLA_CPU_JIT",
                   "GPU": "XLA_GPU_JIT",
                   }

        if isinstance(devices, str):
            devices = [devices]

        for device in devices:
            command = ["tf2xla_supported_ops", f"--device={mapping[device]}"]

            suffix if suffix.startswith(".") else f".{suffix}"
            dst = Path(dst)
            new_stem = "".join((dst.stem, "_", device))
            path_of_file = str(dst.with_stem(new_stem).with_suffix(suffix))

            try:
                with open(path_of_file, "w") as file:
                    subprocess.run(command, stdout=file)
            except Exception:
                print(f"""There was an error creating the markdown table for {device}. \
                Are you sure you sourced your CMSSW enviroment with 'csmenv'?""")
            return path_of_file

    @staticmethod
    def read_markdown_table(table: str) -> dict:
        """
        Reads a *table* created with 'OpsTable.create_markdown_table'.
        The information is turned into a dictionary, which is returned.

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

        with open(table) as file:
            lines = file.read().splitlines(True)

            # filter out the relevenat ops
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
                allowed_types_cpu, allowed_types_gpu = None, None
                if is_cpu:
                    allowed_types_cpu = allowed_types
                elif is_gpu:
                    allowed_types_gpu = allowed_types

                ops[operation] = {"op": operation,
                                  "cpu": is_cpu,
                                  "gpu": is_gpu,
                                  "allowed_types_cpu": allowed_types_cpu,
                                  "allowed_types_gpu": allowed_types_gpu,
                                  }
            return ops

    def merge_markdown_tables(self, *tables: tuple[str]) -> dict:
        """
        Merges multiple tables of different devices, but same TF version into 1 dictionary.

        WARNING: Since its not possible to see from which version the markdown table is generate, try
        to not mix tables from differnt tensorflow versions.

        Returns:
            dict: dictionary of ops with allowed types
        """

        # dont do anything when only 1 table exists
        if len(tables) == 1 | isinstance(tables, str):
            return OpsTable.read_markdown_table(self.table_locations)

        # update multiple tables
        ops_dicts = []
        for table in self.table_locations:
            ops_dicts.append(self.read_markdown_table(table))

        # set first table dictionary
        merged_dict = ops_dicts.pop(-1)
        for ops_dict in ops_dicts:
            for ops_name, value in ops_dict.items():
                # if ops_name exist -> update
                # else -> set value
                if ops_name in merged_dict:
                    for device in self.supported_devices:
                        legit_ops = merged_dict[ops_name][device] | value[device]
                        allowed_types = f'allowed_types_{device}'

                        merged_dict[ops_name][device] = legit_ops

                        # TODO: allowed type change per device?
                else:
                    merged_dict[ops_name] = value
        from IPython import embed
        embed()
        return merged_dict

    @staticmethod
    def print_compare_2_tables(first_table, second_table):
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
        first_table = OpsTable.read_markdown_table(first_table)
        second_table = OpsTable.read_markdown_table(second_table)

        first_keys = first_table.keys()
        second_keys = second_table.table.keys()
        # exclusive or
        or_keys = first_keys ^ second_keys

        # get intersection, left, right exclusive
        same_ops = sorted(list(first_keys & second_keys))
        own_exclusive_ops = sorted(list(first_keys & or_keys))
        other_exclusive_ops = sorted(list(second_keys & or_keys))

        # print everything in a aligned manner
        pack = [("Intersection of Ops", same_ops),
                ("Exclusive to Self", own_exclusive_ops),
                ("Exclusive to Other", other_exclusive_ops)]

        for header, ops in pack:
            print(header.center(40, "*"))
            for op in ops:
                print(op.center(40))
            print("".center(40, "-"))


class ModelLoader():
    def __init__(self, model_location):
        self._model_location = Path(model_location)

        # FLAG set by self.set_flags(*model*)
        self.is_keras = None
        self.is_saved_model = None
        self.is_graph = None
        self._set_flags(self.model_location)  # set is_FLAGS

    @property
    def FLAGS(self):
        return self.is_keras, self.is_saved_model, self.is_graph

    def _set_flags(self, path):
        """
        Sets flags for given model. With this it is easier to check if a model is
        from keras, tf or a graph

        """
        def _is_keras_model(saved_model_dir) -> bool:
            # helper function to check if a model is saved by keras API
            # when this file exists its a keras model
            p = saved_model_dir / "keras_metadata.pb"
            return p.exists()

        path = Path(path)
        if path.is_dir():
            # saved models are dirs.
            self.is_saved_model = tf.saved_model.contains_saved_model(str(path))
            self.is_keras = _is_keras_model(path)
        else:
            self.is_graph = str(path).endswith(".pb")

    def check_model_existance(self, path_to_model):
        # raises errors if path_to_model does no exist, or if this is neither a SavedModel or Graph
        if not Path(path_to_model).exists():
            raise ValueError(f"Path {path_to_model} does not exist.")

        if not self.is_saved_model and not self.is_graph:
            raise Exception(f"""Your chosen directory is neither a SavedModel directory,
            nor a graph *.pb file . Please check if your path is correct:\n {path_to_model}""")

    @property
    def model_location(self):
        return str(self._model_location)

    def load_saved_model(self, path_to_model):
        """
        Returns:
            SavedModel
        """
        # if graph is loaded return None for model
        if self.is_keras:
            return tf.keras.models.load_model(path_to_model)
        else:
            return tf.saved_model.load(path_to_model)

    def load_pb_file(self, path_to_pb, as_graph_def=False) -> tf.compat.v1.GraphDef | tf.Graph:
        """ Reads *path_to_pb* and returns GraphDef of the graph. This function is used to
        load frozen graphs. If <as_grapH_def> is True, a graph_def definition is returned.
        Default is False, thus a loaded Graph is returned.

        Args:
            path_to_pb (str): Location of the pb file

        Returns:
            tf.compat.v1.GraphDef: The loaded GraphDef of the path_to_pb
        """
        with tf.io.gfile.GFile(path_to_pb, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        if as_graph_def:
            return graph_def
        else:
            with tf.compat.v1.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name="")
                return graph

    def load_model(self, path_to_model):
        # check if file exists or is a graph file
        self.check_model_existance(path_to_model)
        if self.is_graph:
            return self.load_pb(path_to_model)
        elif self.is_saved_model:
            return self.load_saved_model(path_to_model)

    @staticmethod
    def convert_pb_2_saved_model(export_dir, graph_pb, input_node_name, output_node_name):
        import tensorflow as tf
        from tensorflow.python.saved_model import signature_constants, tag_constants

        export_dir = './test_models/converted_frouzen_graph'
        graph_pb = '/afs/desy.de/user/w/wiedersb/cmsml/cmsml/to_integrate/tests/test_models/test_freeze_graph.pb'

        builder = tf.compat.v1.saved_model.Builder(export_dir)

        with tf.io.gfile.GFile(graph_pb, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        sigs = {}
        with tf.compat.v1.Session(graph=tf.compat.v1.Graph()) as sess:
            # name="" is important to ensure we don't get spurious prefixing
            tf.import_graph_def(graph_def, name="")
            graph = tf.compat.v1.get_default_graph()
            input_node = graph.get_tensor_by_name("input_0:0")
            output_node = graph.get_tensor_by_name("Identity:0")

            sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
                tf.compat.v1.saved_model.predict_signature_def(
                {"input": input_node},
                {"output": output_node})

            builder.add_meta_graph_and_variables(sess,
                                                 [tag_constants.SERVING],
                                                 signature_def_map=sigs)

        builder.save()

    def get_ports(saved_model, ports=["inputs, outputs"], *, print_only=False):
        # Get the signature defs
        signature_defs = tf.saved_model.get_signature_defs(saved_model)
        if isinstance(ports, str):
            ports = (ports,)
        # Get the input and output tensors
        tensor_names = {}
        for port in ports:
            tensor_name = signature_defs[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs[port].name

            if print_only:
                print(f"Your port {port} has following name: \n tensor_name")
            else:
                tensor_names[port] = tensor_name
        return tensor_names

    def extract_graph_from_saved_model(self, serving_key: str) -> tf.compat.v1.Graph:
        """Extract graph if the model is NOT a keras model.

        Returns:
            tf.SignatureDef: Inference function if model is a keras model otherwise None
        """
        if not self.is_keras_model:
            inference_func = self.model.signatures[serving_key]
            # there is a bug for TFV < 2.8 where python garbage collector kills the model
            # due to having no weak link. This is not a nice solution but it works.
            inference_func._backref_to_saved_model = self.model
            self.graph = inference_func
        else:
            return None


class ModelOpsChecker(ModelLoader):

    def __init__(self, tf_model: str):
        """
        This class taks as input a SavedModel or pb file and returns all unique operators.
        These unique operators are then matched against their XLA compatiblity.

        !!! Be aware, that the compatible list of Ops changes with each tensorflow version. !!!

        Args:
            tf_model (str): path to tf.saved_model
        """
        super().__init__(tf_model)
        self._model_location = Path(tf_model)
        # None if graph is loaded not Savedmodel
        self.loaded_model = self.load_model(self.model_location)

    def get_unique(self, attribute: str, signature: str = None, node_def_number: int = 0) -> set[str, str]:
        """
        Get the attribute of a NodeDef saved unter *signature*. Following options a valid for *attribute*:
        name: Unique string identifier for the node.
        op: String that specifies the type of operation.
        input: List of strings that specify the input tensors for the node.
        device: Optional string that specifies the device on which the operation should be executed
        attr: Map of attribute names to attribute values. Provide additional information about the op

        Args:
            attribute (str): Description
            signature (str, optional): Signature within the graph (default: 'serving_default').
            node_def_number (int, optional): Number of the function def to look up (default: 0)

        Returns:
            set[str, str]: Description

        Raises:
            AttributeError: Raises Error if node_def_number exceeds the number of FunctionDefs.
        """

        # set default signature key
        if signature is None:
            signature = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY



        # get tf.compat.v1.GraphDef

        graph_def = None
        node_def = None
        if self.is_saved_model:
            graph_def = self.loaded_model.signatures[signature].graph.as_graph_def()
        elif self.is_graph:
            graph_def = self.load_pb_file(self.model_location, as_graph_def=True)

        # extract tf.compat.v1.NodeDef from GraphDef.
        try:
            node_def = graph_def.library.function[node_def_number].node_def
        except:
            # raise error when wrong function library is taken.
            num_funcs = len(graph_def.library.function[node_def_number])
            if num_funcs > 1:
                msg = f"You have {num_funcs} FunctionDef's. Change your node_def_number to another value"
                raise AttributeError(msg)

        unique_attributs = {getattr(node, attribute) for node in node_def}
        return unique_attributs

    def get_unique_names(self, signature: str = "serving_default",
                         node_def_number: int = 0) -> set[str, str]:
        """
        Returns a set of all node_names in a graph saved under signature within a tf.saved_model

        """
        return self.get_unique("name", signature, node_def_number)

    def get_unique_ops(self, signature: str = "serving_default",
                    node_def_number: int = 0) -> set[str, str]:
        return self.get_unique("op", signature, node_def_number)

    def find_match(self, signature: str, table_path: str,
                device: str, node_def_number: int = 0) -> list[str, bool]:
        # get unique ops from graph
        ops_in_graph = self.get_unique_ops(signature, node_def_number)

        # get uniqure ops from table
        comptabile_ops_table = OpsTable.read_markdown_table(table_path)
        ops_in_table = OpsTable.get_unique_ops(table=comptabile_ops_table, device=device)

        # compare and check if op is in deed compatible
        compatible_ops_list = [(ops, ops in ops_in_table) for ops in ops_in_graph]
        return compatible_ops_list

    def print_compatible_ops(self, signature: str, table_path: str, device, print_uncompatible_ops=False):
        compatible_list = self.find_match(signature, table_path, device)
        header = ("Operartion", "has XLA")

        print(f"{device.upper().center(40)}")
        print(f"{header[0]:20}|{header[1]:5}")
        print("-" * 20 + "|" + "-" * 20)
        for op, status in compatible_list:
            print(f"{op:20}|{str(bool(status)):5}")

        # check if any are not compatible
        if print_uncompatible_ops:
            uncompatible = [op for op, status in compatible_list if not status]
            print(f"You have following uncompatible ops:", *uncompatible, sep="\n")


if __name__ == "__main__":
    # path_table = "./compat_ops_tables/2_6_4_XLA_CPU_JIT.txt"
    gpu_path_table = "./compat_ops_tables/XLA_GPU_JIT_old.txt"
    cpu_path_table = "./compat_ops_tables/XLA_CPU_JIT_old.txt"

    # gpu_table = OpsTable(gpu_path_table)
    # cpu_table = OpsTable(cpu_path_table)
    b_table = OpsTable(gpu_path_table, cpu_path_table)

    base = "/afs/desy.de/user/w/wiedersb/cmsml/cmsml/to_integrate/tests/test_models/"
    feed_forward_models = {"graph": base + "test_freeze_graph.pb",
                           "keras": base + "test_keras_model", "tf2": base + "test_saved_model"}
    # lstm_path = "/afs/desy.de/user/w/wiedersb/CMSSW_12_4_0/src/PerfTests/aot_convert/hhbtag_lstm/models/saved_model_HHbtag_v1_par_0_static"

    ff_net = ModelOpsChecker(feed_forward_models["keras"])
    # lstm_net = ModelOpsChecker(lstm_path)

    from IPython import embed
    embed()
