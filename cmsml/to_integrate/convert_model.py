# coding: utf-8

from __future__ import annotations


import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf


class ModelConverter(object):

    def __init__(self,
                 src_dir: str,
                 dst_dir: str = "",
                 serving_key: str = "serving_default",
                 batch_sizes: str = "1",
                 ):
        """
        Create an ModelConverter helper instance that exctract all meta information necessary
        to prepare a bare Tensorflow or Keras SavedModel for AOT compilation.

        To start the conversion process call cls.save_static_graph.
        The static graph is saved within the .pb file at *dst* using the
        serving_key: "batch_size_N", with N beeing your *batch_sizes*.

        Args:
            src_dir (str): path of "Keras" or "Tensorflow" saved model dir
            dst_dir (str, optional): Destionation of the static model ".pb file". Default: "", saves
            within src_dir with appended "_static".
            serving_key (str, optional): key under which the src graph is stored
            (default: "serving_default")
            batch_sizes (str, optional): String with batch sizes (delimited by space) or Sequence
            of ints.
        """
        self.src, self.dst = self._check_paths(src_dir, dst_dir)
        self.is_keras_model = self._is_keras(self.src)  # set keras flag

        self.model = self._load_model()  # the actual saved model
        self.serving_key = serving_key
        self.concrete_function = self.model.signatures[serving_key]  # concrete function of the model
        self.batch_sizes = self._map_batch_sizes(batch_sizes)  # create sequence of ints
        self.signatures = None  # the static signatures, after the extration

    @staticmethod
    def colored(s: str, color: str = "white") -> str:
        """Helper function to create colored strings for get better readability

        Args:
            string (str): String you want to have colored
            color (str, optional): "black", "red", "green", "yellow", "blue", "magenta", "white",

        Returns:
            str: String with color coding at start and end.
        """
        color = {
            "black": "90",
            "red": "91",
            "green": "92",
            "yellow": "93",
            "blue": "94",
            "magenta": "95",
            "white": "97",
        }.get(color, "97")

        return "\033[{}m{}\033[0m".format(color, s)

    @staticmethod
    def _print_success(string: str):
        print(ModelConverter.colored(string, "green"))

    @staticmethod
    def _print_notation(string: str):
        print(ModelConverter.colored(string, "yellow"))

    @staticmethod
    def _print_warning(string: str):
        print(ModelConverter.colored(string, "red"))

    def _check_paths(self, src: str, dst: str) -> tuple(Path, Path):
        """Helper function to prepare src and dst paths and provide proper error messages.
        Cases that are covered:
        - directory does no exist
        - src model is not a SavedModel
        - prevent src to be overwritten by static model

        Args:
            src (str): Directory of Tensorflow or Keras SavedModel
            dst (str): Destionation of the static model.

        Returns:
            tuple(Path, Path): Path object of src and dst.
        """
        def is_saved_model(src: str) -> Path:
            """Helper function to if valid saved model exists under *src*
            Args:
                src (str): Path to saved model

            Returns:
                Path: Posix or WindowsPath of path

            Raises:
                ValueError: If path does not exists
            """
            path = Path(src)
            if not path.exists():
                raise ValueError(f"Directory does not exists: \n {str(path)}")
            if not tf.saved_model.contains_saved_model(str(path)):
                raise ValueError(f"Directory: {str(path)} does not contain a tensorflow saved_model")
            return path

        def parse_dst(src: str | Path, dst: str | Path) -> Path:
            """Helper function to append dst with default name and preventing overwriting the source model

            Args:
                src (str): Source of model
                dst (str): Destination of static model

            Returns:
                Path: Path Object of destination of static model
            """
            src = Path(src)
            if dst == "":
                # when no destination is given use alternate source path
                dst = src.with_name("".join((src.name, "_static")))
            else:
                # prevent overwriting src
                if Path(dst) == Path(src):
                    msg = "You can not overwrite your source model. Please choose another *dst*"
                    raise ValueError(msg)
            return Path(dst)
        return is_saved_model(src), parse_dst(src, dst)

    @staticmethod
    def _map_batch_sizes(batch_sizes: str | [list, tuple]):
        """
        Helper function to convert *batch_sizes* argument into a sequence of int's.

        """
        # maps str with space delimiter to tuple with int"s
        # floats are converted to int
        if isinstance(batch_sizes, str):
            # fail if you use wrong delimiter
            try:
                batch_size = tuple(map(int, batch_sizes.split(" ")))
            except:
                raise("Only single space are accepted delimiter for batch sizes")
        elif isinstance(batch_sizes, (tuple, list)):
            batch_size = tuple(map(int, batch_sizes))
        else:
            msg = f"""Batch sizes type is {type(batch_sizes)}, but needs to be
            of type [tuple, list] or str delimited by space.
            """
            raise TypeError(msg)

        # clean up double values, but preserve order
        cleaned_batch_size = tuple(sorted(set(batch_size), key=batch_size.index))
        if len(cleaned_batch_size) != len(batch_size):
            ModelConverter._print_notation("Doubled batch sizes are detected and removed.")
        return cleaned_batch_size

    @staticmethod
    def _is_keras(model_path: Path) -> bool:
        """ Checks if model under *model_path* was saved with:
        tf.saved_model.save or tf.keras.saved_model.save
        Keras models have an additional keras_metadata file within the SavedModel directory.
        Args:
            model_path (Path): Path to Tensorflow or Keras SavedModel.

        Returns:
            bool: True = Model is saved by Keras API, False = Model is saved by TF core API.
        """
        p = model_path.joinpath("keras_metadata.pb")
        keras_model_exists = p.is_file()
        ModelConverter._print_success(f"Model within Path is a Keras Model: {keras_model_exists}")
        return keras_model_exists

    def _load_model(self):
        """Helper function to loads the model using self.src.

        Returns:
            tf.Model" | "tf.keras.model: Loaded tensorflow or keras model
        """
        model_path = str(self.src)
        self._print_notation(f"Load model: {model_path}")
        with tf.device("/cpu:0"):
            if self.is_keras_model:
                name = "Keras"
                model = tf.keras.models.load_model(model_path)
            else:
                name = "Tensorflow"
                model = tf.saved_model.load(model_path)
            self._print_success(f"Succesfully loaded {name} model: {model_path}")
        return model

    def _static_tensorspecs(self, batch_size: int) -> dict[tf.TensorSpec]:
        """
        Extract the input layers information of the loaded model and creates an appropiate
        Tensorspecs with static <batch_size> Shape.

        Args:
            batch_size (int): Integer number, used to create the static shape

        Returns:
            tf.TensorSpec: The with same shape as before, but with static batch size
        """
        static_tensorspecs = {}
        # get input layer information
        input_layers = self.concrete_function.structured_input_signature[1]
        # copy layer information and change batch size
        for name, input_layer in input_layers.items():
            shape = input_layer.shape.as_list()

            # ignore empty argument shapes
            if not shape:
                continue

            # shapes first element is always the batch size
            shape[0] = batch_size
            new_name = f"bs{batch_size}_{input_layer.name}"

            # tensorflows default naming scheme with ":" are problematic in the name argument
            new_name = new_name.replace(":", "_")
            # static tensorspecs are in a dictionary with name of the input layer as key
            # this is necessary since get_concrete_function uses input layers name as keywords args.
            static_tensorspecs[name] = tf.TensorSpec(shape=tf.TensorShape(shape),
                                              dtype=input_layer.dtype,
                                              name=input_layer.name)
        return static_tensorspecs

    def create_static_signatures(self) -> dict[tf.types.experimental.ConcreteFunction]:
        """
        Exctract the concrete function assoziated with a static <batch_size>.
        Takes a Sequence of ints and returns a dictionary with concrete functions.
        The keys of the dictionary follow \"batch_size_<batch_size>\" template.
        """

        signature_dictionary = {}
        # create the concrete functions and fill signature dictionary
        for batch_size in self.batch_sizes:
            save_serving_key = "".join(("batch_size_", str(batch_size)))
            self._print_notation(f"Create signature with servingkey: {save_serving_key}")
            # create static tensorspec to create concrete function
            static_tensorspec = self._static_tensorspecs(batch_size=batch_size)
            # exctract concrete function with shape of tensorspec
            signature_dictionary[save_serving_key] = tf.function(
                self.concrete_function).get_concrete_function(**static_tensorspec)

        self.signatures = signature_dictionary
        return signature_dictionary

    def save_signatures(self):
        """
        Save signature as saved_model under <saving_dir>
        """
        self._print_notation(f"Start saving signatures at {str(self.dst)}:")
        tf.saved_model.save(self.model, str(self.dst),
                            signatures=self.signatures)
        print_saving_dir = self.colored(str(self.dst), "red")
        self._print_success(f"Saved the model with static signatures at: {print_saving_dir}")

    def save_static_graph(self):
        self.create_static_signatures()
        self.save_signatures()

    def set_device(self, model, serving_key, device="/device:CPU:*"):
        """
        Copy models graph, but with cleared device information.
        """

        graph = model.signatures[serving_key].graph

        # removes all device placement information
        for op in graph.get_operations():
            op._set_device = device
        return graph


def main(
    src: str,
    dst: str = "",
    batch_sizes: tuple[int] = (1,),
    serving_key: str = "serving_default",
):
    ModelConverter(
        src_dir=src,
        dst_dir=dst,
        batch_sizes=batch_sizes,
        serving_key=serving_key,
    ).save_static_graph()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="convert TensorFlow models to an AOT compatible format",
        prog="cmsml_convert_aot",
    )

    parser.add_argument(
        "--src",
        required=True,
        help="path of the tensorFlow saved model directory",
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="output path for the converted model",
    )
    parser.add_argument(
        "--batch-sizes",
        "-b",
        default=(1,),
        type=(lambda s: tuple(map(int, s.strip().split(",")))),
        help="comma-separated list of batch sizes to convert the model for; default: 1",
    )
    parser.add_argument(
        "--serving-key",
        "-k",
        default="serving_default",
        help="serving key of the model in --src; default: serving_default",
    )

    args = parser.parse_args()
    main(**args.__dict__)
