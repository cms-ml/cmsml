import sys
sys.path.append("..")
import tensorflow as tf

from aot_compat_test import ModelManipulation, GraphOpsChecker
import unittest
from pathlib import Path

class TestModelManipulation(unittest.TestCase):
    def setUp(self):

        test_dir = Path('test_models')
        self.saved_keras_model = str(test_dir / 'test_keras_model')
        self.saved_tf_model = str(test_dir / 'test_saved_model')
        self.saved_graph = str(test_dir / 'test_freeze_graph.pb')

        self.mm_graph = ModelManipulation(self.saved_graph)
        self.mm_keras_model = ModelManipulation(self.saved_keras_model)
        self.mm_tf_model = ModelManipulation(self.saved_tf_model)


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
            mm = ModelManipulation('non_existing')
            mm.check_model_existance()

        # should raise Exception for existing directories that are no tf.SavedModel
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp_dir:
            with self.assertRaises(Exception):
                mm = ModelManipulation(tmp_dir)
                mm.check_model_existance()

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


if __name__ == '__main__':
    unittest.main()
