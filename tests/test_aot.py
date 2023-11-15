# coding: utf-8

from __future__ import annotations

import os
import functools
import subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cmsml
from cmsml.util import tmp_dir, tmp_file
# from cmsml.tensorflow.aot import get_graph_ops, OpsData

from . import CMSMLTestCase


# check if the tf2xla_supported_ops command exists
p = subprocess.run("type tf2xla_supported_ops", shell=True)
HAS_TF2XLA_SUPPORTED_OPS = p.returncode == 0


def skip_if_no_tf2xla_supported_ops(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not HAS_TF2XLA_SUPPORTED_OPS:
            return
        return func(*args, **kwargs)
    return wrapper


class AOTTestCase(CMSMLTestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._tf = None
        self._tf1 = None
        self._tf_version = None

    @property
    def tf(self):
        if self._tf is None:
            self._tf, self._tf1, self._tf_version = cmsml.tensorflow.import_tf()
        return self._tf

    @property
    def tf1(self):
        if self._tf1 is None:
            self._tf, self._tf1, self._tf_version = cmsml.tensorflow.import_tf()
        return self._tf1

    @property
    def tf_version(self):
        if self._tf_version is None:
            self._tf, self._tf1, self._tf_version = cmsml.tensorflow.import_tf()
        return self._tf_version

    def create_graph_def(self, create="saved_model", **kwargs):
        import cmsml.tensorflow.tools as cmsml_tools

        # helper function to create GraphDef from SavedModel or Graph
        tf = self.tf

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(10,), dtype=tf.float32, name="input"))
        model.add(tf.keras.layers.BatchNormalization(axis=1, renorm=True))
        model.add(tf.keras.layers.Dense(100, activation="tanh"))
        model.add(tf.keras.layers.BatchNormalization(axis=1, renorm=True))
        model.add(tf.keras.layers.Dense(3, activation="softmax", name="output"))

        if create == "saved_model":
            with tmp_dir(create=False) as keras_path, tmp_dir(create=False) as tf_path:

                tf.saved_model.save(model, tf_path)
                model.save(keras_path, overwrite=True, include_optimizer=False)

                default_signature = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
                tf_graph_def = cmsml_tools.load_graph_def(tf_path, default_signature)
                keras_graph_def = cmsml_tools.load_graph_def(keras_path, default_signature)
            return tf_graph_def, keras_graph_def

        elif create == "graph":
            concrete_func = tf.function(model).get_concrete_function(tf.ones((2, 10)))

            with tmp_file(suffix=".pb") as pb_path:
                cmsml_tools.save_graph(pb_path, concrete_func, variables_to_constants=False)
                graph_graph_def = cmsml.tensorflow.load_graph_def(
                    pb_path,
                    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
                )
            return graph_graph_def

    @skip_if_no_tf2xla_supported_ops
    def test_get_graph_ops_saved_model(self):
        from cmsml.tensorflow.aot import get_graph_ops

        tf_graph_def, keras_graph_def = self.create_graph_def(create="saved_model")

        graph_ops = set(get_graph_ops(tf_graph_def, node_def_number=0))
        expected_ops = {
            "AddV2", "BiasAdd", "Const", "Identity", "MatMul", "Mul", "NoOp", "Rsqrt", "Softmax",
            "Sub", "Tanh",
        }
        io_ops = {"ReadVariableOp", "Placeholder"}

        ops_without_io = graph_ops - io_ops
        self.assertSetEqual(ops_without_io, expected_ops)

    @skip_if_no_tf2xla_supported_ops
    def test_get_graph_ops_graph(self):
        from cmsml.tensorflow.aot import get_graph_ops

        concrete_function_graph_def = self.create_graph_def(create="graph")
        graph_ops = set(get_graph_ops(concrete_function_graph_def, node_def_number=0))

        expected_ops = {
            "AddV2", "BiasAdd", "Const", "Identity", "MatMul", "Mul", "NoOp", "Rsqrt", "Softmax",
            "Sub", "Tanh",
        }
        io_ops = {"ReadVariableOp", "Placeholder"}

        ops_without_io = graph_ops - io_ops
        self.assertSetEqual(ops_without_io, expected_ops)


class OpsTestCase(CMSMLTestCase):

    def __init__(self, *args, **kwargs):
        super(OpsTestCase, self).__init__(*args, **kwargs)

        self._tf = None
        self._tf1 = None
        self._tf_version = None

    @property
    def tf(self):
        if self._tf is None:
            self._tf, self._tf1, self._tf_version = cmsml.tensorflow.import_tf()
        return self._tf

    @property
    def tf1(self):
        if self._tf1 is None:
            self._tf, self._tf1, self._tf_version = cmsml.tensorflow.import_tf()
        return self._tf1

    @property
    def tf_version(self):
        if self._tf_version is None:
            self._tf, self._tf1, self._tf_version = cmsml.tensorflow.import_tf()
        return self._tf_version

    @skip_if_no_tf2xla_supported_ops
    def test_parse_ops_table(self):
        from cmsml.tensorflow.aot import OpsData

        ops_dict = OpsData.parse_ops_table(device="cpu")
        expected_ops = ("Abs", "Acosh", "Add", "Atan", "BatchMatMul", "Conv2D")

        # check if ops name and content exist
        # since content changes with every version only naiv test is done

        for op in expected_ops:
            self.assertTrue(bool(ops_dict[op]["allowed_types"]))

    @skip_if_no_tf2xla_supported_ops
    def test_determine_ops(self):
        from cmsml.tensorflow.aot import OpsData

        # function to merge multiple tables
        devices = ("cpu", "gpu")

        ops_data = OpsData(devices)
        ops_data_ops = ops_data.ops
        # for these ops cpu and gpu implentation are guaranteed
        expected_ops = ("Abs", "Acosh", "Add", "Atan", "BatchMatMul", "Conv2D")

        # content for cpu and gpu should not be empty
        for op in expected_ops:
            for device in devices:
                self.assertTrue(bool(ops_data_ops[op][device]))
