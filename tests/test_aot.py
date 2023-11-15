from __future__ import annotations
import os

import cmsml
from cmsml.util import tmp_dir, tmp_file
import cmsml.tensorflow.tools as cmsml_tools
from cmsml.tensorflow.aot import get_graph_ops, OpsData
from . import CMSMLTestCase


class AotTestCase(CMSMLTestCase):

    def __init__(self, *args, **kwargs):
        super(AotTestCase, self).__init__(*args, **kwargs)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

    def create_graph_def(self, create='saved_model', **kwargs):
        # helper function to create GraphDef from SavedModel or Graph
        tf = self.tf

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(10,), dtype=tf.float32, name="input"))
        model.add(tf.keras.layers.BatchNormalization(axis=1, renorm=True))
        model.add(tf.keras.layers.Dense(100, activation="tanh"))
        model.add(tf.keras.layers.BatchNormalization(axis=1, renorm=True))
        model.add(tf.keras.layers.Dense(3, activation="softmax", name="output"))

        if create == 'saved_model':
            with tmp_dir(create=False) as keras_path, tmp_dir(create=False) as tf_path:

                tf.saved_model.save(model, tf_path)
                model.save(keras_path, overwrite=True, include_optimizer=False)

                default_signature = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
                tf_graph_def = cmsml_tools.load_graph_def(tf_path, default_signature)
                keras_graph_def = cmsml_tools.load_graph_def(keras_path, default_signature)
            return tf_graph_def, keras_graph_def

        elif create == 'graph':
            concrete_func = tf.function(model).get_concrete_function(tf.ones((2, 10)))

            with tmp_file(suffix=".pb") as pb_path:
                cmsml_tools.save_graph(pb_path, concrete_func, variables_to_constants=False)
                graph_graph_def = cmsml.tensorflow.load_graph_def(pb_path,
                                                 tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
            return graph_graph_def

    def test_get_graph_ops_saved_model(self):
        tf_graph_def, keras_graph_def = self.create_graph_def(create='saved_model')

        graph_ops = set(get_graph_ops(tf_graph_def, node_def_number=0))
        expected_ops = {'AddV2',
                        'BiasAdd',
                        'Const',
                        'Identity',
                        'MatMul',
                        'Mul',
                        'NoOp',
                        'Rsqrt',
                        'Softmax',
                        'Sub',
                        'Tanh'
                        }

        io_ops = {'ReadVariableOp', 'Placeholder'}
        ops_without_io = graph_ops - io_ops
        self.assertSetEqual(ops_without_io, expected_ops)

    def test_get_graph_ops_graph(self):
        concrete_function_graph_def = self.create_graph_def(create='graph')
        graph_ops = set(get_graph_ops(concrete_function_graph_def, node_def_number=0))

        expected_ops = {'AddV2',
                        'BiasAdd',
                        'Const',
                        'Identity',
                        'MatMul',
                        'Mul',
                        'NoOp',
                        'Rsqrt',
                        'Softmax',
                        'Sub',
                        'Tanh'
                        }

        io_ops = {'ReadVariableOp', 'Placeholder'}

        ops_without_io = graph_ops - io_ops
        self.assertSetEqual(ops_without_io, expected_ops)


class OpsTestCase(CMSMLTestCase):

    def __init__(self, *args, **kwargs):
        super(OpsTestCase, self).__init__(*args, **kwargs)

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

    def test_parse_ops_table(self):
        ops_dict = OpsData.parse_ops_table(device='cpu')
        expected_ops = ('Abs', 'Acosh', 'Add', 'Atan', 'BatchMatMul', 'Conv2D')

        # check if ops name and content exist
        # since content changes with every version only naiv test is done

        for op in expected_ops:
            self.assertTrue(bool(ops_dict[op]['allowed_types']))

    def test___determine_ops(self):
        # function to merge multiple tables
        devices = ('cpu', 'gpu')

        ops_data = OpsData(devices)
        ops_data_ops = ops_data.ops
        # for these ops cpu and gpu implentation are guaranteed
        expected_ops = ('Abs', 'Acosh', 'Add', 'Atan', 'BatchMatMul', 'Conv2D')

        # content for cpu and gpu should not be empty
        for op in expected_ops:
            for device in devices:
                self.assertTrue(bool(ops_data_ops[op][device]))
