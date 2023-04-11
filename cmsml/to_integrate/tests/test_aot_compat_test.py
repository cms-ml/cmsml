from __future__ import annotations
import sys
sys.path.append("..")
sys.path.append("../..")
import tensorflow as tf


import unittest
from pathlib import Path
from util import tmp_dir, tmp_file
from check_aot_compatibility import OpsTable, ModelLoader, ModelOpsChecker


class TestOpsTable(unittest.TestCase):
    CPU_2_6_4 = "compat_ops_tables/2_6_4_XLA_CPU_JIT.txt"
    CPU_TABLE = "compat_ops_tables/XLA_CPU_JIT_old.txt"
    GPU_TABLE = "compat_ops_tables/XLA_GPU_JIT_old.txt"

    def setUp(self):
        self.table = OpsTable()

    def test_create_markdown_table(self):
        with tmp_file() as path:
            file_dst = Path(path) / "test_table"
            created_file_path = self.table.create_markdown_table(dst=file_dst,
                                        devices="CPU")

            self.assertTrue(Path(created_file_path).exists)

            # TODO GPU currently not supported by CMMSW, remove Raises when implemented
            with self.assertRaises(NotImplementedError):
                file_path = self.create_markdown_table(dst=file_dst,
                                        devices="GPU")
                self.assertTrue(Path(file_path).exists)

    def test_read_markdown_table(self):
        # read a table and check if 1 entry is right interpreted
        ops_dict_cpu = self.table.read_markdown_table(self.CPU_TABLE)
        test_entry_cpu = {
            'Abs': {
                'op': 'Abs',
                'cpu': True,
                'gpu': False,
                'allowed_types_cpu': 'T={double,float,int32,int64}',
                'allowed_types_gpu': None,
            },
        }

        self.assertEqual(test_entry_cpu['Abs'], ops_dict_cpu['Abs'])

        # repeat for GPU to see if switch works
        ops_dict_gpu = self.table.read_markdown_table(self.GPU_TABLE)
        test_entry_gpu = {
            'Abs': {
                'op': 'Abs',
                'cpu': False,
                'gpu': True,
                'allowed_types_cpu': None,
                'allowed_types_gpu': 'T={double,float,int32,int64}',
            },
        }

        self.assertEqual(test_entry_gpu['Abs'], ops_dict_gpu['Abs'])

    @unittest.skip("TODO Function not fully implemented")
    def test_merge_markdown_tables(self):
        # combine 2 tables information.
        # used to get overview of all possible devices.
        ops_dict = self.table.merge_markdown_tables(self.CPU_TABLE, self.GPU_TABLE)
        test_entry_merged = {
            'Abs': {
                'op': 'Abs',
                'cpu': True,
                'gpu': True,
                'allowed_types_cpu': 'T={double,float,int32,int64}',
                'allowed_types_gpu': 'T={double,float,int32,int64}',
            },
        }
        print('\n' * 3)

        for k in test_entry_merged['Abs'].keys():
            print(test_entry_merged['Abs'][k], ops_dict['Abs'][k])

        print('\n' * 3)
        self.assertEqual(test_entry_merged['Abs'], ops_dict['Abs'])

    @unittest.skip("TODO Not sure what to test to be honest")
    def test_get_unique_ops(self):
        cpu_ops_dict = self.table.read_markdown_table(self.CPU_TABLE)
        gpu_ops_dict = self.table.read_markdown_table(self.GPU_TABLE)

        all_ops = self.table.get_get_unique_ops()

        with self.assertRaises(Exception):
            # raise if non-supported op is given
            not_supported_ops = self.table.get_get_unique_ops(all_ops, "dummy_device")


class TestModelLoader(unittest.TestCase):
    def setUp(self):

        test_dir = Path('test_models')
        self.saved_keras_model = str(test_dir / 'test_keras_model')
        self.saved_tf_model = str(test_dir / 'test_saved_model')
        self.saved_graph = str(test_dir / 'test_freeze_graph.pb')

        self.mm_graph = ModelLoader(self.saved_graph)
        self.mm_keras_model = ModelLoader(self.saved_keras_model)
        self.mm_tf_model = ModelLoader(self.saved_tf_model)

    def create_dummy_model(self):
        x1 = tf.keras.Input(shape=(2,), name="foo")
        x2 = tf.keras.Input(shape=(2,), name="bar")
        x = tf.concat([x1, x2], axis=1)
        a1 = tf.keras.layers.Dense(10, activation="elu")(x)
        y = tf.keras.layers.Dense(2, activation="softmax")(a1)
        model = tf.keras.Model(inputs=[x1, x2], outputs=y)
        return model

    def test_set_flags(self):
        def saved_model(mm):
            self.assertTrue(mm.is_keras)
            self.assertTrue(mm.is_saved_model)
            self.assertFalse(mm.is_graph)

        def graph_model(mm):
            self.assertFalse(mm.is_keras)
            self.assertFalse(mm.is_saved_model)
            self.assertTrue(mm.is_graph)


    def test_check_model_existance(self):
        with self.assertRaises(ValueError):
            mm = ModelLoader('non_existing')
            mm.check_model_existance(mm.model_location)

        # should raise Exception for existing directories that are no tf.SavedModel
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp_dir:
            with self.assertRaises(Exception):
                mm = ModelLoader(tmp_dir)
                mm.check_model_existance(mm.model_location)

    def test_load_saved_model(self):
        l_keras_model = self.mm_keras_model.load_saved_model(self.mm_keras_model.model_location)
        l_tf_model = self.mm_tf_model.load_saved_model(self.mm_tf_model.model_location)

        self.assertIsInstance(l_keras_model, tf.keras.Model)
        # instance of tensorflow api models is kinda hard to get
        # instead check if isSavedModel and is not keras model
        self.assertNotIsInstance(l_tf_model, tf.keras.Model)
        self.assertTrue(tf.saved_model.contains_saved_model(self.mm_tf_model.model_location))


    def test_load_pb(self):
        l_graph = self.mm_graph.load_pb(self.mm_graph.model_location)
        self.assertIsInstance(l_graph, tf.compat.v1.Graph)

    def test_load_model(self):
        loaded_keras_model = self.mm_keras_model.load_model(self.mm_keras_model.model_location)
        self.assertIsInstance(loaded_keras_model, tf.keras.Model)
        loaded_tf_model = self.mm_tf_model.load_model(self.mm_tf_model.model_location)
        self.assertIsInstance(loaded_tf_model, tf.keras.Model)

class TestModelOpsChecker(unittest.TestCase):
    def setUp(self):

        test_dir = Path('test_models')
        self.saved_keras_model = str(test_dir / 'test_keras_model')
        self.saved_tf_model = str(test_dir / 'test_saved_model')
        self.saved_graph = str(test_dir / 'test_freeze_graph.pb')

        self.graph_model_ops = ModelOpsChecker(self.saved_graph)
        self.keras_model_ops = ModelOpsChecker(self.saved_keras_model)
        self.tf_model_ops = ModelOpsChecker(self.saved_tf_model)

        self.default_signature = 'serving_default'


    def test_get_graph_def(self):
        # should return graph def of savedmodel and graph

        graph_graph_def = self.graph_model_ops.get_graph_def(signature=self.default_signature)
        tf_graph_def = self.tf_model_ops.get_graph_def(signature=self.default_signature)
        self.assertEqual(graph_graph_def, tf_graph_def)

    def test_get_node_def(self):
        pass


if __name__ == '__main__':
    unittest.main()
