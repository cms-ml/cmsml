# coding: utf-8

"""
TensorFlow tools.
"""

from __future__ import annotations

__all__ = []

import os
import warnings
from types import ModuleType
from typing import Any
from tensorflow.core.framework.graph_pb2 import GraphDef

from cmsml.util import MockModule


tf = MockModule("tensorflow")

tf_cpp_log_levels = {
    "DEBUG": 0,
    "INFO": 1,
    "WARNING": 2,
    "ERROR": 3,
}


def import_tf(
    log_level: int | str = "WARNING",
    autograph_verbosity: int = 3,
) -> tuple[ModuleType, ModuleType | None, tuple[int]]:
    """
    Imports TensorFlow and returns a 3-tuple containing the module itself, the v1 compatibility
    API (i.e. the TensorFlow module itself if v1 is the primarily installed version), and the
    package version as a 3-tuple containing integers. Example:

    .. code-block:: python

        tf, tf1, tf_version = import_tf()

    At some point in the future, when v1 support might get fully removed from TensorFlow 2 or
    higher, the second tuple element might be *None*.

    The verbosity of logs printed by TensorFlow and AutoGraph can be controlled through *log_level*
    and *autograph_verbosity*.
    """
    # set the TF_CPP_MIN_LOG_LEVEL before tf gets imported
    if log_level in tf_cpp_log_levels:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(tf_cpp_log_levels[log_level])

    import tensorflow as tf

    # set log and verbosity levels
    if log_level:
        tf.get_logger().setLevel(log_level)
    if autograph_verbosity >= 0:
        tf.autograph.set_verbosity(autograph_verbosity)

    # split the version into three parts
    tf_version = tuple(map(int, tf.__version__.split(".", 2)))

    # keep a reference to the v1 API as long as v2 provides compatibility
    tf1 = None
    if tf_version[0] == 1:
        tf1 = tf
    elif getattr(tf, "compat", None) and getattr(tf.compat, "v1", None):
        tf1 = tf.compat.v1

    return tf, tf1, tf_version


def save_graph(
    path: str,
    obj: Any,
    variables_to_constants: bool = False,
    output_names: list[str] | None = None,
    *args,
    **kwargs,
) -> None:
    """
    Deprecated. Please use :py:func:`save_frozen_graph`.
    """
    warnings.warn(
        "save_graph() is deprecated, please use save_frozen_graph() instead",
        DeprecationWarning,
    )
    return save_frozen_graph(
        path,
        obj,
        variables_to_constants=variables_to_constants,
        output_names=output_names,
        *args,
        **kwargs,
    )


def save_frozen_graph(
    path: str,
    obj: Any,
    variables_to_constants: bool = False,
    output_names: list[str] | None = None,
    *args,
    **kwargs,
) -> None:
    """
    Extracts a TensorFlow graph from an object *obj* and saves it at *path*. The graph is optionally
    transformed into a simpler representation with all its variables converted to constants when
    *variables_to_constants* is *True*. The saved file contains the graph as a protobuf. The
    accepted types of *obj* greatly depend on the available API versions.

    When the v1 API is found (which is also the case when ``tf.compat.v1`` is available in v2),
    ``Graph``, ``GraphDef`` and ``Session`` objects are accepted. However, when
    *variables_to_constants* is *True*, *obj* must be a session and *output_names* should refer
    to names of operations whose subgraphs are extracted (usually just one).

    For TensorFlow v2, *obj* can also be a compiled keras model, or either a polymorphic or
    concrete function as returned by ``tf.function``. Polymorphic functions either must have a
    defined input signature (``tf.function(input_signature=(...,))``) or they must accept no
    arguments in the first place. See the TensorFlow documentation on `concrete functions
    <https://www.tensorflow.org/guide/concrete_function>`__ for more info.

    *args* and *kwargs* are forwarded to ``tf.train.write_graph`` (v1) or ``tf.io.write_graph``
    (v2).
    """
    tf, tf1, tf_version = import_tf()
    path = os.path.expandvars(os.path.expanduser(str(path)))
    graph_dir, graph_name = os.path.split(path)

    # default as_text value
    kwargs.setdefault("as_text", path.endswith((".pbtxt", ".pb.txt")))

    # convert keras models and polymorphic functions to concrete functions, v2 only
    if tf_version[0] != 1:
        from tensorflow.python.keras.saving import saving_utils
        from tensorflow.python.eager.def_function import Function
        from tensorflow.python.eager.function import ConcreteFunction

        if isinstance(obj, tf.keras.Model):
            learning_phase_orig = tf.keras.backend.get_value(tf.keras.backend.learning_phase())
            tf.keras.backend.set_learning_phase(False)
            model_func = saving_utils.trace_model_call(obj)
            if model_func.function_spec.arg_names and not model_func.input_signature:
                raise ValueError(
                    "when obj is a keras model callable accepting arguments, its "
                    "input signature must be frozen by building the model",
                )
            obj = model_func.get_concrete_function()
            tf.keras.backend.set_learning_phase(learning_phase_orig)

        elif isinstance(obj, Function):
            if obj.function_spec.arg_names and not obj.input_signature:
                raise ValueError(
                    "when obj is a polymorphic function accepting arguments, its ",
                    "input signature must be frozen")
            obj = obj.get_concrete_function()

    # convert variables to constants
    if variables_to_constants:
        if tf1 and isinstance(obj, tf1.Session):
            if not output_names:
                raise ValueError(
                    "when variables_to_constants is true, output_names must "
                    f"contain operations to export, got '{output_names}' instead",
                )
            obj = tf1.graph_util.convert_variables_to_constants(
                obj,
                obj.graph.as_graph_def(),
                output_names,
            )

        elif tf_version[0] != 1:
            from tensorflow.python.framework import convert_to_constants

            if not isinstance(obj, ConcreteFunction):
                raise TypeError(
                    "when variables_to_constants is true, obj must be a concrete "
                    f"or polymorphic function, got '{obj}' instead",
                )
            obj = convert_to_constants.convert_variables_to_constants_v2(obj)

        else:
            raise TypeError(
                f"cannot convert variables to constants for object '{obj}', type not "
                f"understood for TensorFlow version {tf.__version__}",
            )

    # extract the graph
    if tf1 and isinstance(obj, tf1.Session):
        graph = obj.graph
    elif tf_version[0] != 1 and isinstance(obj, ConcreteFunction):
        graph = obj.graph
    else:
        graph = obj

    # write it
    if tf_version[0] == 1:
        tf1.train.write_graph(graph, graph_dir, graph_name, *args, **kwargs)
    else:
        tf.io.write_graph(graph, graph_dir, graph_name, *args, **kwargs)


def load_graph(
    path: str,
    create_session: bool | None = None,
    session_kwargs: dict | None = None,
    as_text: bool | None = None,
) -> tf.Graph | tuple[tf.Graph, tf.Session]:
    """
    Deprecated. Please use :py:func:`load_frozen_graph`.
    """
    warnings.warn(
        "load_graph() is deprecated, please use load_frozen_graph() instead",
        DeprecationWarning,
    )
    return load_frozen_graph(
        path=path,
        create_session=create_session,
        session_kwargs=session_kwargs,
        as_text=as_text,
    )


def load_frozen_graph(
    path: str,
    create_session: bool | None = None,
    session_kwargs: dict | None = None,
    as_text: bool | None = None,
) -> tf.Graph | tuple[tf.Graph, tf.Session]:
    """
    Reads a saved TensorFlow graph from *path* and returns it. When *create_session* is *True*,
    a session object (compatible with the v1 API) is created and returned as the second value of
    a 2-tuple. The default value of *create_session* is *True* when TensorFlow v1 is detected,
    and *False* otherwise. In case a session is created, *session_kwargs* are forwarded to the
    session constructor as keyword arguments when set. When *as_text* is either *True* or *None*,
    and the file extension is ``".pbtxt"`` or ``".pb.txt"``, the content of the file at *path* is
    expected to be a human-readable text file. Otherwise, it is read as a binary protobuf file.
    Example:

    .. code-block:: python

        graph = load_frozen_graph("path/to/model.pb", create_session=False)

        graph, session = load_frozen_graph("path/to/model.pb", create_session=True)
    """
    tf, tf1, tf_version = import_tf()
    path = os.path.expandvars(os.path.expanduser(str(path)))

    # default create_session value
    if create_session is None:
        create_session = tf_version[0] == 1
    if create_session and not tf1:
        raise NotImplementedError(
            "the v1 compatibility layer of TensorFlow v2 is missing, "
            "but required by when create_session is True",
        )

    # default as_text value
    if as_text is None:
        as_text = path.endswith((".pbtxt", ".pb.txt"))

    graph = tf.Graph()
    with graph.as_default():
        graph_def = graph.as_graph_def()

        if as_text:
            # use a simple pb reader to load the file into graph_def
            from google.protobuf import text_format
            with open(path, "rb") as f:
                text_format.Merge(f.read(), graph_def)

        else:
            # use the gfile api depending on the TF version
            if tf_version[0] == 1:
                from tensorflow.python.platform import gfile
                with gfile.FastGFile(path, "rb") as f:
                    graph_def.ParseFromString(f.read())
            else:
                with tf.io.gfile.GFile(path, "rb") as f:
                    graph_def.ParseFromString(f.read())

        # import the graph_def (pb object) into the actual graph
        tf.import_graph_def(graph_def, name="")

    if create_session:
        session = tf1.Session(graph=graph, **(session_kwargs or {}))
        return graph, session
    else:
        return graph


def load_graph_def(
    model_path: str,
    serving_key: str = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
) -> GraphDef:
    """
    Loads the model saved at *model_path* and returns the GraphDef of it. Supported input types are tensorflow and keras
    SavedModels, as well as frozen graphs.
    """
    tf, tf1, tf_version = import_tf()

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
        # loaded_saved_model.signatures[serving_key].function_def.node_def
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
    Load and return the SavedModel stored at *model_path*. If the model was saved using keras it will be loaded using
    keras SavedModel API, otherwise tensorflow's SavedModel API is used.
    """
    tf, tf1, tf_version = import_tf()

    model_path = os.path.expandvars(os.path.expanduser(str(model_path)))

    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "keras_metadata.pb")):
        model = tf.keras.models.load_model(model_path)
    else:
        model = tf.saved_model.load(model_path)

    return model


def write_graph_summary(
    graph: tf.Graph,
    summary_dir: str,
    **kwargs,
) -> None:
    """
    Writes the summary of a *graph* to a directory *summary_dir* using a ``tf.summary.FileWriter``
    (v1) or ``tf.summary.create_file_writer`` (v2). This summary can be used later on to visualize
    the graph via tensorboard. *graph* can be either a graph object or a path to a protobuf file. In
    the latter case, :py:func:`load_frozen_graph` is used and all *kwargs* are forwarded.

    .. note::
        When used with TensorFlow v1, eager mode must be disabled.
    """

    # prepare the summary dir
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    # read the graph when a string is passed
    if isinstance(graph, str):
        graph = load_frozen_graph(graph, create_session=False, **kwargs)

    # further handling is version dependent
    tf, tf1, tf_version = import_tf()
    if tf_version[0] == 1:
        # switch to non-eager mode for the FileWriter to work
        eager = getattr(tf1, "executing_eagerly", lambda: False)()
        if eager:
            tf1.disable_eager_execution()

        # write to file
        writer = tf1.summary.FileWriter(summary_dir)
        writer.add_graph(graph)

        # reset the eager mode
        if eager:
            tf1.enable_eager_execution()

    else:  # 2.X
        from tensorflow.python.ops import summary_ops_v2 as summary_ops

        # create the writer
        writer = tf.summary.create_file_writer(summary_dir)

        # write the graph
        with writer.as_default():
            summary_ops.graph(graph.as_graph_def())

        # close
        writer.close()
