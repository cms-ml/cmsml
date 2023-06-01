from __future__ import annotations
import sys
sys.path.append('..')
sys.path.append('../..')
import tensorflow as tf


import unittest
from pathlib import Path
from util import tmp_dir, tmp_file
import cmsml.tensorflow.tools as cmsml_tools
from check_aot_compatibility import OpsTable, ModelOpsChecker


class TestOpsTable(unittest.TestCase):
    CPU_TABLE = 'tests/compat_ops_tables/XLA_CPU_JIT_old.txt'
    GPU_TABLE = 'tests/compat_ops_tables/XLA_GPU_JIT_old.txt'

    def setUp(self):
        self.table = OpsTable(self.CPU_TABLE, self.GPU_TABLE)

    @unittest.skip
    def test_create_markdown_table(self):
        with tmp_file() as path:
            file_dst = Path(path) / 'test_table'
            created_file_path = self.table.create_markdown_table(
                device='CPU', save_location=file_dst, suffix='.md')

            # save markdown file works
            self.assertTrue(Path(created_file_path).exists)

            # TODO GPU currently not supported by CMMSW, remove Raises when implemented
            with self.assertRaises(NotImplementedError):
                file_path = self.create_markdown_table(dst=file_dst,
                                        devices='GPU')
                self.assertTrue(Path(file_path).exists)

    def test_read_markdown_table(self):
        # read a table and check if 1 entry is right interpreted
        ops_dict_cpu = self.table.read_markdown_table(self.CPU_TABLE)
        test_entry_cpu = {
            'Abs': {
                'op': 'Abs',
                'cpu': 'T={double,float,int32,int64}',
                'gpu': None,
            },
        }

        self.assertEqual(test_entry_cpu['Abs'], ops_dict_cpu['Abs'])

        # repeat for GPU to see if switch works
        ops_dict_gpu = self.table.read_markdown_table(self.GPU_TABLE)
        test_entry_gpu = {
            'Abs': {
                'op': 'Abs',
                'gpu': 'T={double,float,int32,int64}',
                'cpu': None,
            },
        }

        self.assertEqual(test_entry_gpu['Abs'], ops_dict_gpu['Abs'])

    def test_merge_markdown_tables(self):
        # combine 2 tables information.
        # used to get overview of all possible devices.
        test_entry_merged = {
            'Abs': {
                'op': 'Abs',
                'cpu': 'T={double,float,int32,int64}',
                'gpu': 'T={double,float,int32,int64}',
            },
            '_XLASend': {'op': '_XLASend',
            'cpu': 'T={bool,complex64,double,float,int32,int64,uint32,uint64}',
            'gpu': None},
        }

        self.assertEqual(test_entry_merged['Abs'], self.table.ops_dict['Abs'])
        self.assertEqual(test_entry_merged['_XLASend'], self.table.ops_dict['_XLASend'])


class TestModelOpsChecker(unittest.TestCase):
    CPU_TABLE = TestOpsTable.CPU_TABLE
    GPU_TABLE = TestOpsTable.GPU_TABLE

    @classmethod
    def setUpClass(cls):
        # global paths to the temp dirs containing the test models
        cls.paths = TestModelOpsChecker.save_test_model()
        cls.serving_key = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY

        cls.graph_model_ops = ModelOpsChecker(
            model_path=TestModelOpsChecker.paths['frozen_graph'], serving_key=cls.serving_key, node_def_number=0)
        cls.keras_model_ops = ModelOpsChecker(
            model_path=TestModelOpsChecker.paths['keras'], serving_key=cls.serving_key, node_def_number=0)
        cls.tf_model_ops = ModelOpsChecker(
            model_path=TestModelOpsChecker.paths['tf'], serving_key=cls.serving_key, node_def_number=0)

    @staticmethod
    def save_test_model():
        model = TestModelOpsChecker.create_dummy_model()
        # save model in tmp as keras, tf2 saved_model and frozen graph
        with tmp_dir(create=False, delete=False) as path_tf:
            tf.saved_model.save(model, path_tf)
        with tmp_dir(create=False, delete=False) as path_keras:
            model.save(path_keras, overwrite=True, include_optimizer=False)
        with tmp_dir(create=False, delete=False) as path_graph:
            path_graph = Path(path_graph) / 'frozen_graph.pb'
            cmsml_tools.save_graph(path=str(path_graph), obj=model, variables_to_constants=True,)

        paths = {"keras": path_keras, "tf": path_tf, "frozen_graph": path_graph}
        return paths

    @staticmethod
    def create_dummy_model():
        x1 = tf.keras.Input(shape=(2,), name='foo')
        x2 = tf.keras.Input(shape=(2,), name='bar')
        x = tf.concat([x1, x2], axis=1)
        a1 = tf.keras.layers.Dense(10, activation='elu')(x)
        y = tf.keras.layers.Dense(2, activation='softmax')(a1)
        model = tf.keras.Model(inputs=[x1, x2], outputs=y)
        return model

    def test_load_graph_def(self):
        # extract graph_def from keras, tf and frozen graphs
        graph_graph_def = self.graph_model_ops.load_graph_def(
            model_path=self.graph_model_ops.model_path, serving_key=self.serving_key)
        tf_graph_def = self.tf_model_ops.load_graph_def(
            model_path=self.tf_model_ops.model_path, serving_key=self.serving_key)
        keras_graph_def = self.keras_model_ops.load_graph_def(
            model_path=self.keras_model_ops.model_path, serving_key=self.serving_key)
        potential_graph_defs = [graph_graph_def, tf_graph_def, keras_graph_def]
        self.assertTrue(all([isinstance(graph_def, tf.compat.v1.GraphDef)
                        for graph_def in potential_graph_defs]))

    def test_unique_ops(self):
        # return unique ops of graph
        should_have_ops = {'BiasAdd',
                         'ConcatV2',
                         'Const',
                         'Elu',
                         'Identity',
                         'MatMul',
                         'NoOp',
                         'Softmax'}

        # ReadVariableOp = Op to read out variable value
        tf_keras_exclusive_ops = {'ReadVariableOp'}
        # PlaceHolder = Placeholder for input tensors of graph
        # in this case: placeholders have name 'foo', 'bar'
        frozen_graph_exclusive_ops = {'Placeholder'}
        tf_keras_ops = should_have_ops.union(tf_keras_exclusive_ops)
        frozen_graph_ops = should_have_ops.union(frozen_graph_exclusive_ops)

        self.assertEqual(self.tf_model_ops.unique_ops, tf_keras_ops)
        self.assertEqual(self.keras_model_ops.unique_ops, tf_keras_ops)
        self.assertEqual(self.graph_model_ops.unique_ops, frozen_graph_ops)

    def test_find_match(self):
        # should return list with tuples(Op, bool(exist in compatiblity table))
        result_base = {('Elu', True),
                       ('Softmax', True),
                       ('MatMul', True),
                       ('Const', True),
                       ('BiasAdd', True),
                       ('Identity', True),
                       ('ConcatV2', True)}

        result_tf_keras_matching = result_base.union({('ReadVariableOp', True)})

        tf_matching = self.tf_model_ops.find_match(self.CPU_TABLE, "cpu")
        keras_matching = self.keras_model_ops.find_match(self.CPU_TABLE, "cpu")
        graph_matching = self.graph_model_ops.find_match(self.CPU_TABLE, "cpu")

        self.assertEqual(tf_matching, result_tf_keras_matching)
        self.assertEqual(keras_matching, result_tf_keras_matching)
        # NoOp and Placeholder are filtered out
        # XLA does not have compatiblity with these dummy ops.
        self.assertEqual(graph_matching, result_base)


if __name__ == '__main__':
    unittest.main()
