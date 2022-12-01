import tensorflow as tf
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Ops():

    def __init__(self):
        self.ops_dict = {}

    def get_unique(self, property: str) -> set[str]:
        """
        Filters the entrys of "self.ops_dict" after < property >.
        Returns only unique entries by using a Set.

        Args:
            property (str): 'op' for internal operation name,
                            'name' to just get the name of the op (most used case)
                            'file' to get the files where the kernel was found

        Returns:
            set[str]: Unique entries of "self.ops_dict[< property >]"
        """
        return {value_from_dict[property] for value_from_dict in self.ops_dict.values()}

    @property
    def unique_ops(self):
        # get unique operations
        return self.get_unique('op')

    @property
    def unique_name(self):
        # get unique names
        return self.get_unique('name')

    @property
    def unique_files(self):
        # get unique used files
        return self.get_unique('file')

    @property
    def unique_cpu(self):
        # TODO filter results after CPU only
        # get unique XLA compatible results for CPU only
        return self.get_unique('cpu')

    @property
    def unique_gpu(self):
        # TODO filter results after GPU only
        # # get unique XLA compatible results for GPU only
        return self.get_unique('gpu')

    @property
    def unique_tpu(self):
        # TODO filter results after TPU only
        # # get unique XLA compatible results for TPU only
        return self.get_unique('tpu')

    def get_location(self, location):
        if location.exists():
            return str(location)
        else:
            raise NameError(f'{str(location)} does not exist.')


class OpsTable(Ops):
    """ This class is used to filter the markdown table with compatible XLA ops generated
        with bazel when compiling tensorflow.
    """

    def __init__(self, table_dir: str):
        """
        Args:
            table_dir (str): Path to the markdown table
        """
        super().__init__()
        self._table_location = Path(table_dir)
        self.main()

    @property
    def table_location(self) -> str:
        # return table location, raise error if missing
        return self.get_location(self._table_location)

    def read_markdown_table(self) -> dict:
        """
        Fills the self.ops_dict with all the information filtered form the markdown table
        located at self.table_location
        """
        with open(self.table_location) as file:
            all_lines = file.read().splitlines(True)
            without_header = all_lines[4:-6]
            for line in without_header:
                cleaned_line = line.replace('`', '').replace(' ', '').replace('\n', '')
                operation, types = cleaned_line.split('|')
                self.ops_dict[operation] = {'file': None, 'name': operation, 'op': operation,
                                            'CPU': False, 'GPU': False, 'TPU': False, 'allowed_types': types}

    def main(self):
        self.read_markdown_table()


class OpsFiles(Ops):
    """ This class is used to search all kernels of tensorflow
        for REGISTER_XLA_OP occurences. This function is used
        by tensorflow to mark an operation as comptabible for XLA usage.

        If you have already a table of compatible ops use the Ops_Table class
        instead.

        !!! Currently this method is not taking into account preprocessor information.
        Therefore a distinguish of CPU, GPU, TPU and OS is not possible.
        Have this in mind when using this class !!!
    """

    def __init__(self, cpp_tensorflow_dir: str = None):
        super().__init__()
        self._tf_location = Path(cpp_tensorflow_dir)
        self.main()

    @property
    def tf2xla_location(self) -> Path:
        """
        Location of the code to compile the kernels used by tensorflow

        Returns:
            Path: Path Object of the kernel path
        """
        return self.get_location(self._tf_location / "tensorflow/compiler/tf2xla/kernels")

    @property
    def tf_location(self) -> str:
        # tensorflow location as string
        return self.get_location(self._tf_location)

    def _filter_op(self, full_operation: str, file_name: str):
        """
        Function filters the actual information from the CPP line of code.

        Args:
            full_operation (str): is a line of CPP code seperated by ";"
            file_name (str): file name where the functions is coming from
        """
        split = full_operation.split(',')
        name = split[0]  # REGISTER_XLA_OP(Name("OPERATION_NAME").else
        operation = split[-1]  # "NAMEOp");

        name = name.replace('REGISTER_XLA_OP(Name("', '').split('"')[0]
        operation = operation.replace(');', '')

        # TODO implement CPU, GPU, TPU tag, THEY ARE SET TO FALSE AS DUMMY

        self.ops_dict[name] = {'file': file_name, 'name': name, 'op': operation,
                               'CPU': False, 'GPU': False, 'TPU': False}  # todo findout how to find this out

    def filter_ops(self, bucket: dict):
        """
        Fills "self.dict_ops" with the actual XLA-compatibility information.

        Args:
            bucket (dict): Dictionary (bucket) with the data coming from "self.read_ops"
            The data is saved using the name of the operation, not the file name!
        """
        for file_name, ops in bucket:
            for op in ops:
                self._filter_op(op, file_name)

    def read_ops(self) -> dict:
        """
        Functions reads the "*.cc" files within the kernel directory.
        This function also prepares the lines

        Returns:
            dict: Dictionary containing filtered data.

        """
        all_cpp_files = Path(self.tf2xla_location).glob('*.cc')  # get all kernel files

        bucket = []
        # subprocess_command = "sed -n '/^REGISTER_XLA_OP(*[;,]*/,/;\n/p ' {path}".format(path = p)    #sed equivalent

        for cpp_file in all_cpp_files:
            with open(cpp_file) as file:
                lines = file.read().split(';')  # split into different command blocks
                f_lines = [code_block.replace(' ', '').replace('\n', '')
                           for code_block in lines]  # remove white space and linebreaks
                # find OPS
                ops = [code_block +
                       ';' for code_block in f_lines if code_block.startswith('REGISTER_XLA_OP(')]
                bucket.append((str(cpp_file.name), ops))  # bundle found place and ops list
        return bucket

    def main(self):
        operations_bucket = self.read_ops()
        self.filter_ops(operations_bucket)


class GraphOpsChecker():

    def __init__(self, tf_model_dir: str):
        """
        This class taks as input a SavedModel or pb file and returns all unique operators.
        These unique operators are then matched against their XLA compatiblity.

        !!! Be aware, that the compatible list of Ops changes with each tensorflow version. !!!

        Args:
            tf_model_dir (str): path to tf.saved_model
        """
        self.model_location = Path(tf_model_dir)
        self.is_keras_model = self._is_keras_model()  # True if model is saved with keras
        self.model = self.load_model()  # the model with all signatures

    def _is_keras_model(self) -> None:
        p = self.model_location / 'keras_metadata.pb'
        self.is_keras_model = p.is_file()

    def load_model(self):
        if tf.saved_model.contains_saved_model(self.model_location):
            if self.is_keras_model:
                model = tf.keras.models.load_model(self.model_location)
            else:
                model = tf.saved_model.load(self.model_location)
            return model
        else:
            raise Exception(f'There is no saved model under the directory: {self.model_location}')

    def _get_node_def(self, signature: str) -> "google.protobuf.pyext._message.RepeatedCompositeContainer":
        # get graph_def
        graph = self.model.signatures[signature].graph.as_graph_def()
        # get node_def,
        # library is list-like "tensorflow.core.framework.function_pb2.FunctionDefLibrary"
        node_def = graph.library.function[0].node_def
        return node_def

    def get_unique_names(self, signature):
        """
        Returns a set of all node_names in a graph saved under signature within a tf.saved_model

        Args:
            signature (str): Name of the signature within the graph.
            If signature is unknown call and your loaded model is called m.
            signarue = m.signatures

        Returns:
            TYPE: Description
        """
        return {node.name for node in self._get_node_def(signature)}

    def get_unique_ops(self, signature='serving_default'):
        return {node.op for node in self._get_node_def(signature)}

    def extract_graph(self, serving_key):
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

    def _extract_graph_keras_model(self, keras_model, inputs=None):
        """
        Args:
            keras_model (TYPE): Your loaded keras model (NOT tf.Saved_Model)
            inputs (None, optional): Use this for complex inputs. Default (get shape from input name tag)
        """
        if inputs is not None:
            for input_layer in self.inputs:
                shape = input_layer.shape.as_list()
                dummy_input = tf.zeros(shape)

         # model.predict_function is only populated after 1 call to predict,
         # dummy input is necessary to ensure this.

        model = 1
        # we want to get the ops necessary for prediction.
        # Only important for AOT, since no training possible
        keras_model.predict(tf.constant([1.0]))
        graph = model.predict_function.get_concrete_function(
            iter([tf.constant([1.0])])).graph  # The concrete function takes an iterator
        isinstance(graph, tf.Graph)  # True

    def check_compatibility(self, signature, compatible_ops_table):
        unique_ops = self.get_unique_ops(signature)
        compatible_ops_list = []
        for ops in unique_ops:
            is_compatible = ops in compatible_ops_table
            compatible_ops_list.append((ops, is_compatible))
        return compatible_ops_list

    def print_compatible_ops(self, table):
        header = ('Operartion', 'has XLA')

        print(f'{header[0]:20}|{header[1]:5}')
        print('-' * 20 + '|' + '-' * 20)
        for op, status in table:
            print(f'{op:20}|{str(bool(status)):5}')

    def read_graphdef(self, pb_file):
        with tf.io.gfile.GFile(pb_file, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def


if __name__ == '__main__':

    path_to_tensorflow = '/Users/bogdanwiederspan/Desktop/tensorflow_repo/tensorflow'
    path_to_table = '/Users/bogdanwiederspan/Documents/docker_mounts/bazel/mount_point.txt'

    table_ops = OpsTable(path_to_table)
    files_ops = OpsFiles(path_to_tensorflow)

    bucket = None
    model_dir = './saved_model_12_128'
    model_dir = '/afs/desy.de/user/w/wiedersb/CMSSW_12_4_0/src/PerfTests/aot_convert/hhbtag_lstm/models/saved_model_HHbtag_v1_par_0_static'
    ops_checker = GraphOpsChecker(model_dir)
#    ops_checker.check_compatibility('batch_size_1', bucket)

    from IPython import embed
    embed()
    exit()
