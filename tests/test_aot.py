from __future__ import annotations
import sys
sys.path.append('..')
sys.path.append('../..')
import tensorflow as tf


import unittest
from pathlib import Path
from util import tmp_dir, tmp_file
import cmsml.tensorflow.tools as cmsml_tools

from aot import get_graph_ops, OpsData


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


    def create_graph_def(self,create='saved_model' **kwargs):
        # helper function to create GraphDef from SavedModel or Graph
        if create == 'saved_model':
            model = self.tf.keras.Sequential()

            model.add(self.tf.keras.layers.InputLayer(input_shape=(10,), dtype=self.tf.float32, name="input"))
            model.add(self.tf.keras.layers.BatchNormalization(axis=1, renorm=True))
            model.add(self.tf.keras.layers.Dense(100, activation="tanh"))
            model.add(self.tf.keras.layers.BatchNormalization(axis=1, renorm=True))
            model.add(self.tf.keras.layers.Dense(3, activation="softmax", name="output"))

            with tmp_dir(create=False) as keras_path, tmp_dir(create=False) as tf_path:
                self.tf.saved_model.save(model, tf_path)
                model.save(keras_path, overwrite=True, include_optimizer=False)

                tf_graph_def, keras_graph_def = cmsml_tools.load_graph_def(tf_path), cmsml_tools.load_graph_def(keras_path)
            return tf_graph_def, keras_graph_def
        elif create == 'graph':
            concrete_func = self.create_tf_function(concrete=True)

            with tmp_file(suffix=".pb") as pb_path:
                cmsml.tensorflow.save_graph(pb_path, concrete_func, variables_to_constants=False)
            return cmsml.tensorflow.load_graph_def(pb_path, self.tf2.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY)


    def test_get_graph_ops_saved_model(self):
        tf_graph_def, keras_graph_def = self.create_graph_def(create='saved_model')

        self.assertRaises(AttributeError,cmsml.tensorflow.get_graph_ops(tf_graph_def, node_def_number=len(graph_def.library.function)+1))

        graph_ops = set(cmsml.tensorflow.get_graph_ops(tf_graph_def, node_def_number=0))
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

        io_ops = {'ReadVariableOp',
                  'Placeholder'
                  }

        ops_without_io = graph_ops - io_ops
        self.assertEqual(ops_without_io, expected_ops)


    def test_get_graph_ops_graph(self):
        concrete_function_graph_def = self.create_graph_def(create='graph')
        graph_ops = set(cmsml.tensorflow.get_graph_ops(concrete_function_graph_def, node_def_number=0))

        expected_ops = {
         'MatMul',
         'AddV2',
         'Tanh',
         'Identity',
         'NoOp'
         }

         io_ops = {'ReadVariableOp', 'Placeholder'}

         ops_without_io = graph_ops - io_op
         self.assertEqual(ops_without_io, expected_ops)




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
        # function to read in the table
        ops_dict = OpsData.parse_ops_table(device='cpu')

        should_have_ops = {'Abs': {'name': 'Abs',
                          'device': 'cpu',
                          'allowed_types': 'T={double,float,int32,int64}'},
                          'Acosh': {'name': 'Acosh',
                          'device': 'cpu',
                          'allowed_types': 'T={complex64,double,float}'},
                          'Add': {'name': 'Add',
                          'device': 'cpu',
                          'allowed_types': 'T={complex64,double,float,int32,int64}'},
                          }

        for should_have_op in should_have_ops.keys():
            self.assertEqual(ops_dict[should_have_op]['allowed_types'], should_have_ops[should_have_op]['allowed_types'])

    def test___determine_ops(self):
        # function to merge multiple tables
        devices = ('cpu','gpu')

        ops_data = OpsData(devices)

        should_have_ops = { 'Abs': {'cpu': 'T={double,float,int32,int64}',
                                  'gpu': 'T={double,float,int32,int64}'
                                  },
                            'Acosh': {'cpu': 'T={complex64,double,float}',
                                  'gpu': 'T={complex64,double,float}'
                                  },
                            'Add': {'cpu': 'T={complex64,double,float,int32,int64}',
                                  'gpu': 'T={complex64,double,float,int32,int64}'
                                  }
                          }
        for op in should_have_ops.keys():
            for device in devices:
                self.assertEqual(should_have_ops[op][device], ops_data._ops_dict[op][device])








