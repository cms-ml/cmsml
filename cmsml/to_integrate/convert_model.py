from __future__ import annotations
from pathlib import Path
import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class ModelConverter():

    def __init__(self,
                 src_dir: str,
                 dst_dir: str = "",
                 serving_key: str = "serving_default",
                 batch_sizes: str = "1",
                 inputs = "input:0"
                 ):
        """
        Create an ModelConverter instance which holds all meta information necessary to
        make a graph static. To start the conversion process call the instance.
        The static graph is saved within the .pb at *dst* using the
        serving_key: "batch_size_N", with N beeing your *batch_sizes*.

        Args:
            src_dir (str): path of "Keras" or "Tensorflow" saved model dir
            dst_dir (str, optional): Destionation of the static model. Default: "", saves
            within src_dir with appended _static.
            serving_key (str, optional): key under which the src models graph is stored
            (default: "serving_default")
            batch_sizes (str, optional): String with batch sizes (delimited by space)
            the network should convert to
        """
        self.src = self._is_saved_model(src_dir)
        self.dst = self._parse_dst(self.src, dst_dir)
        self.serving_key = serving_key
        self.batch_sizes = self._map_batch_sizes(batch_sizes)  # create sequence of ints
        self.is_keras_model = self._is_keras(self.src)  # set keras flag
        self.model = self.load_model()  # the actual saved model
        self.graph = self._extract_graph()  # the concrete function (graph) of the model
        self.signatures = None  # the static signatures, after the extration
        self.input_names = (inputs,) if isinstance(inputs,str) else inputs

    @staticmethod
    def _pretty_color(string: str, color: str = "white") -> str:
        """Helper function to create colored strings for get better readability

        Args:
            string (str): String you want to have colored
            color (str, optional): "black", "red", "green", "yellow", "blue", "magenta", "white",

        Returns:
            str: String with color coding at start and end.
        """
        START = "\033["
        RESET = f"{START}0m"

        COLORS = {
            "black": "90",
            "red": "91",
            "green": "92",
            "yellow": "93",
            "blue": "94",
            "magenta": "95",
            "white": "97",
        }
        return "".join((START, COLORS.get(color, "97"), "m", string, RESET))

    @staticmethod
    def _print_success(string: str):
        print(ModelConverter._pretty_color(string, "green"))

    @staticmethod
    def _print_notation(string: str):
        print(ModelConverter._pretty_color(string, "yellow"))

    @staticmethod
    def _print_warning(string: str):
        print(ModelConverter._pretty_color(string, "red"))

    @staticmethod
    def _is_saved_model(path: str) -> Path:
        """Checks if saved model exists under path, returns a Path Object.
        To actually get the str use str(Path).

        Args:
            path (str): Path to saved model

        Returns:
            Path: Posix or WindowsPath of path

        Raises:
            ValueError: If path does not exists
        """
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Directory does not exists: \n {str(path)}")
        if not tf.saved_model.contains_saved_model(str(path)):
            raise ValueError(f"Directory: {str(path)} does not contain a tensorflow saved_model")
        return path

    @staticmethod
    def _parse_dst(src: str, dst: str) -> Path:
        """Summary

        Args:
            src (str): Description
            dst (str): Description

        Returns:
            Path: Description
        """
        if dst == "":
            # when no destination is given use alternate source
            dst = src.with_name("".join((src.name, "_static")))
        else:
            if dst == "same":
                dst = src
        return Path(dst)

    @staticmethod
    def _map_batch_sizes(batch_sizes: str | [list, tuple]):
        # maps str with space delimiter to tuple with int"s
        # floats are converted to int
        if isinstance(batch_sizes, str):
            # will fail if you use wrong delimiter
            bs = tuple(map(int, batch_sizes.split(" ")))
        elif isinstance(batch_sizes, (tuple, list)):
            bs = tuple(map(int, batch_sizes))
        else:
            msg = f"""Batch sizes type is {type(batch_sizes)}, but needs to be
            of type [tuple, list] or str which is delimited by space.
            """
            raise ValueError(msg)

        # clean up double values, but preserve order
        cleaned_bs = tuple(sorted(set(bs), key=bs.index))
        if len(cleaned_bs) != len(bs):
            ModelConverter._print_notation("Doubled batch sizes are detected and removed.")
        return cleaned_bs

    @staticmethod
    def _is_keras(model_path: Path) -> bool:
        """ Checks if model under <model_path> was saved with:
        tf.saved_model.save or tf.keras.saved_model.save
        This is done by checking if a keras_metadata file exists within the saved model dir.

        Args:
            model_path (Path): Path to model.

        Returns:
            bool: True = Model is saved by Keras API, False = Model is saved by TF core API.
        """
        p = model_path.joinpath("keras_metadata.pb")
        keras_model_exists = p.is_file()
        ModelConverter._print_success(f"Model within Path is a Keras Model: {keras_model_exists}")
        return keras_model_exists

    @property
    def inputs(self) -> list[str]:
        """Get the inputs of the loaded model/graph as Sequence of ints

        Returns:
            Sequence[str]: List with input names
        """
        # if self.input_names:
        #     return {for }

        # we are only interested in the tensorspecs of the inputs
        if self.is_keras_model:
            # keras inputs function returns correct order of inputs!
            return [input.type_spec for input in self.model.inputs]
        else:
            # all other serialized node information are not ordered!
            # _arg_keywords, sadly returns unordered input names
            # TODO: search right way.
            # overwrite order by own names
            correct_ordered_inputs = self.input_names if self.input_names else self.graph._arg_keywords


            # tuple with dict with unsorted input nodes: NODE_NAME:TensorSpec
            # first entry is empty
            graph_inputs = self.graph.structured_input_signature[1]
            return [graph_inputs[input_node_name] for input_node_name in correct_ordered_inputs]


    def load_model(self):
        """Loads the model using self.src.

        Returns:
            tf.Model" | "tf.keras.model: Loaded tensorflow or keras model
        """
        model_path = str(self.src)
        self._print_notation(f"Load model: {model_path}")
        if self.is_keras_model:
            name = "Keras"
            model = tf.keras.models.load_model(model_path)
        else:
            name = "Tensorflow"
            model = tf.saved_model.load(model_path)
        self._print_success(f"Succesfully loaded {name} model")
        return model

    def _extract_graph(self) -> tf.SignatureDef:
        """Extract graph if the model is NOT a keras model.

        Returns:
            tf.SignatureDef: Inference function if model is a keras model otherwise None
        """
        if not self.is_keras_model:
            inference_func = self.model.signatures[self.serving_key]
            # there is a bug for TFV < 2.8 where python garbage collector kills the model
            # due to having no weak link. This is not a nice solution but it works.
            inference_func._backref_to_saved_model = self.model
            return inference_func
        else:
            return None

    def _static_tensorspecs(self, batch_size: int) -> tf.TensorSpec:
        """
        Takes the input layers of loaded model or graph and creates a
        Tensorspecs with static <batch_size> Shape and new name (batch_size_<batch_size>).

        Args:
            batch_size (int): A number, used to create the static shape

        Returns:
            tf.TensorSpec: The now static shape
        """
        tensorspecs = []
        for input_layer in self.inputs:
            shape = input_layer.shape.as_list()

            # ignore empty argument shapes
            if not shape:
                continue
            # shapes first element is always the batch size
            shape[0] = batch_size
            new_name = "_".join(
                ("batch_size", str(batch_size), input_layer.name))

            # networks : are problematic in name
            new_name = new_name.replace(":", "_")

            layer_description = tf.TensorSpec(shape=tf.TensorShape(shape),
                                              dtype=input_layer.dtype,
                                              name=input_layer.name)
            tensorspecs.append(layer_description)
        return tensorspecs

    def _static_concrete_function(self, batch_size: int) -> tf.types.experimental.ConcreteFunction:
        """
        Exctract the concrete function assoziated with the static <batch_size>.

        Args:
            batch_size (int): Description

        Returns:
            tf.types.experimental.ConcreteFunction: Description
        """
        # Creates a static tensorspec and use it to extract the corresponding concrete function
        static_tensorspec = self._static_tensorspecs(batch_size=batch_size)

        # get the concrete function for the signature: static_tensorspec
        new_signature = tf.function(
            self.model.__call__).get_concrete_function(inputs=static_tensorspec, training=False, mask=None)
        return new_signature

    def create_static_signatures(self):
        """
        Takes a Sequence of ints and returns a dictionary with concrete functions.
        The keys of the dictionary follow \"batch_size_<batch_size>\" template.
        The concrete functions have static batch_size.
        """
        signature_dictionary = {}

        # create the concrete functions and fill signature dictionary
        for bs in self.batch_sizes:
            save_serving_key = "".join(("batch_size_", str(bs)))
            self._print_notation(f"Create signature with servingkey: {save_serving_key}")

            signature = self._static_concrete_function(bs)
            signature_dictionary[save_serving_key] = signature

        # save signature dictionary
        self.signatures = signature_dictionary

    def decap_create_static_signatures(self):
        # works for 1 node input only
        """
        Takes a Sequence of ints and returns a dictionary with concrete functions.
        The keys of the dictionary follow \"batch_size_<batch_size>\" template.
        The concrete functions have static batch_size.
        """
        signature_dictionary = {}

        # create the concrete functions and fill signature dictionary
        for bs in self.batch_sizes:
            save_serving_key = "".join(("batch_size_", str(bs)))
            self._print_notation(f"Create signature with servingkey: {save_serving_key}")

            signature = self._static_concrete_function(bs)
            signature_dictionary[save_serving_key] = signature

        # save signature dictionary
        self.signatures = signature_dictionary

    def clear_devices(self, graph_def):
        """
        Create new graph with settings of graph_def, but cleared device.
        """
        # removes all device placement information
        for node in graph_def.node:
            node.device = ""

        # create create modified graph
        graph = tf.Graph()
        with graph:
            # load cleared graph_def into new graph
            tf.graph_util.import_graph_def(graph_def,
                                           name="")
        return graph

    def save_signatures(self):
        """
        Save signature as saved_model under <saving_dir>
        """

        self._print_notation(f"Starting save of signatures at {str(self.dst)}:")

        tf.saved_model.save(self.model, str(self.dst),
                            signatures=self.signatures)

        print_saving_dir = self._pretty_color(str(self.dst), "red")
        self._print_success(f"Saved the model with static signatures at: {print_saving_dir}")

    def __call__(self):
        self.create_static_signatures()
        self.save_signatures()


def main(src: str,
         dst: str = "",
         batch_sizes: str = "1",
         serving_key: str = "serving_default"):
    converter = ModelConverter(src, dst,
                               serving_key=serving_key,
                               batch_sizes=batch_sizes)
    converter()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="""
                    Tool to set batch size dimension to a static
                    value and save it in a saved model graph with a keyword of name: batch_size_N,
                    with N being your given batch size
                    """,
                                     prog="Prog")

    parser.add_argument("--src",
                        help="""Path of the tensorflow saved model dir.""",
                        type=str,
                        required=True,
                        dest="src")

    parser.add_argument("--dst",
                        help="""Destination path of the resulting static saved model.
                        Following special cases are allowed:\n
                        \t - dst="" (default): the graph is saved in the same
                        directory as the source, with the suffix "_static".\n
                        \t - dst="replace" the graph is saved within the src TF_saved_model\n
                        \t - Anything else is just a interpreted as path to the saved_model dir""",
                        type=str,
                        required=True,
                        dest="dst")

    parser.add_argument("--serving_key",
                        help="""
                        A *.pb file can habit multiple graphs, each with its own unique serving_key.
                        The default is "serving_default". """,
                        type=str,
                        default="serving_default",
                        required=False,
                        dest="serving_key")

    parser.add_argument("-b",
                        "--batch_sizes",
                        help=""" Pass a string with ints using spaces as delimiter, e.g. "1 2 4"
                        will create a model with static batch sizes 1 2 and 4.
                        This sets the suffix of the serving key under which the mode is saved.
                        From our example the model with batch size 1 has the key: batch_size_1.^""",
                        type=str,
                        required=True,
                        dest="batch_sizes")

    args = parser.parse_args()
    main(**args.__dict__)
