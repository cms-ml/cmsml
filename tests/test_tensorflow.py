# coding: utf-8

"""
TensorFlow tests.
"""

import os
import unittest

import cmsml
from cmsml.util import tmp_file, tmp_dir


class TensorFlowTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TensorFlowTestCase, self).__init__(*args, **kwargs)

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.tf, self.tf1, self.tf_version = cmsml.tensorflow.import_tf()

        self._W = None
        self._b = None

    @property
    def W(self):
        if self._W is None:
            self._W = self.tf.Variable(self.tf.ones([10, 1]))
        return self._W

    @property
    def b(self):
        if self._b is None:
            self._b = self.tf.Variable(self.tf.ones([1]))
        return self._b

    def create_keras_model(self, tf):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.InputLayer(input_shape=(10,), dtype=tf.float32, name="input"))
        model.add(tf.keras.layers.BatchNormalization(axis=1, renorm=True))
        model.add(tf.keras.layers.Dense(100, activation="tanh"))
        model.add(tf.keras.layers.BatchNormalization(axis=1, renorm=True))
        model.add(tf.keras.layers.Dense(3, activation="softmax", name="output"))

        return model

    def create_tf1_session(self, graph):
        tf = self.tf1
        if tf is None:
            return None

        return tf.Session(graph=graph, config=tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
            device_count={"GPU": 0},
        ))

    def create_tf1_graph(self, create_session=True):
        tf = self.tf1
        if tf is None:
            return None, None

        graph = tf.Graph()
        with graph.as_default():
            x_ = tf.placeholder(tf.float32, [None, 10], name="input")
            W = tf.Variable(tf.ones([10, 1]))
            b = tf.Variable(tf.ones([1]))
            h = tf.add(tf.matmul(x_, W), b)
            y = tf.tanh(h, name="output")  # noqa

        if not create_session:
            return graph

        session = self.create_tf1_session(graph)

        with graph.as_default():
            session.run(tf.global_variables_initializer())

        return graph, session

    def create_tf_function(self, frozen=False, no_input=False, concrete=False):
        tf = self.tf

        if frozen:
            # polymorphic function, frozen input signature
            @tf.function(input_signature=(tf.TensorSpec(shape=[2, 10], dtype=tf.float32),))
            def func(x):
                h = tf.add(tf.matmul(x, self.W), self.b)
                y = tf.tanh(h, name="output")
                return y

        elif no_input:
            # polymorphic function, empty input signature
            @tf.function
            def func():
                x = tf.ones([2, 10])
                h = tf.add(tf.matmul(x, self.W), self.b)
                y = tf.tanh(h, name="output")
                return y

        else:
            # polymorphic function, unknown input signature
            @tf.function
            def func(x):
                h = tf.add(tf.matmul(x, self.W), self.b)
                y = tf.tanh(h, name="output")
                return y

            if concrete:
                # convert to concrete function with known signature
                func = func.get_concrete_function(tf.TensorSpec(shape=[2, 10], dtype=tf.float32))

        return func

    def test_import_tf(self):
        tf, tf1, tf_version = cmsml.tensorflow.import_tf()

        self.assertEqual(len(tf_version), 3)

        if tf_version[0] == "1":
            self.assertEqual(tf, tf1)

    def test_save_graph(self):
        graph, session = self.create_tf1_graph()
        if graph is None or session is None:
            return

        with tmp_file(suffix=".pb") as path:
            cmsml.tensorflow.save_graph(path, graph, variables_to_constants=False)
            self.assertTrue(os.path.exists(path))

        with tmp_file(suffix=".pb.txt") as path:
            cmsml.tensorflow.save_graph(path, graph, variables_to_constants=False)
            self.assertTrue(os.path.exists(path))

        with tmp_file(suffix=".pb") as path:
            cmsml.tensorflow.save_graph(path, graph.as_graph_def(), variables_to_constants=False)
            self.assertTrue(os.path.exists(path))

        with tmp_file(suffix=".pb") as path:
            cmsml.tensorflow.save_graph(path, session, variables_to_constants=False)
            self.assertTrue(os.path.exists(path))

        with tmp_file(suffix=".pb") as path:
            cmsml.tensorflow.save_graph(path, session, variables_to_constants=True,
                output_names=["output"])
            self.assertTrue(os.path.exists(path))

        with tmp_file(suffix=".pb") as path:
            with self.assertRaises(ValueError):
                cmsml.tensorflow.save_graph(path, session, variables_to_constants=True)
            self.assertFalse(os.path.exists(path))

    def test_save_polymorphic_function_error(self):
        poly_func = self.create_tf_function()

        with self.assertRaises(ValueError):
            with tmp_file(suffix=".pb") as path:
                cmsml.tensorflow.save_graph(path, poly_func, variables_to_constants=False)

        with self.assertRaises(ValueError):
            with tmp_file(suffix=".pb") as path:
                cmsml.tensorflow.save_graph(path, poly_func, variables_to_constants=True)

    def test_save_empty_polymorphic_function(self):
        empty_poly_func = self.create_tf_function(no_input=True)

        with tmp_file(suffix=".pb") as path:
            cmsml.tensorflow.save_graph(path, empty_poly_func, variables_to_constants=False)
            self.assertTrue(os.path.exists(path))

        with tmp_file(suffix=".pb") as path:
            cmsml.tensorflow.save_graph(path, empty_poly_func, variables_to_constants=True)
            self.assertTrue(os.path.exists(path))

    def test_save_frozen_polymorphic_function(self):
        frozen_poly_func = self.create_tf_function(frozen=True)

        with tmp_file(suffix=".pb") as path:
            cmsml.tensorflow.save_graph(path, frozen_poly_func, variables_to_constants=False)
            self.assertTrue(os.path.exists(path))

        with tmp_file(suffix=".pb") as path:
            cmsml.tensorflow.save_graph(path, frozen_poly_func, variables_to_constants=True)
            self.assertTrue(os.path.exists(path))

    def test_save_concrete_function(self):
        concrete_func = self.create_tf_function(concrete=True)

        with tmp_file(suffix=".pb") as path:
            cmsml.tensorflow.save_graph(path, concrete_func, variables_to_constants=False)
            self.assertTrue(os.path.exists(path))

        with tmp_file(suffix=".pb") as path:
            cmsml.tensorflow.save_graph(path, concrete_func, variables_to_constants=True)
            self.assertTrue(os.path.exists(path))

    def test_save_keras_model_v1(self):
        model = self.create_keras_model(self.tf1)

        with tmp_file(suffix=".pb") as path:
            cmsml.tensorflow.save_graph(path, model, variables_to_constants=False)
            self.assertTrue(os.path.exists(path))

        with tmp_file(suffix=".pb") as path:
            cmsml.tensorflow.save_graph(path, model, variables_to_constants=True)
            self.assertTrue(os.path.exists(path))

        with tmp_file(suffix=".pb") as path:
            cmsml.tensorflow.save_graph(path, self.tf1.keras.backend.get_session(),
                variables_to_constants=False)
            self.assertTrue(os.path.exists(path))

    def test_save_keras_model_v2(self):
        model = self.create_keras_model(self.tf)

        with tmp_file(suffix=".pb") as path:
            cmsml.tensorflow.save_graph(path, model, variables_to_constants=False)
            self.assertTrue(os.path.exists(path))

        with tmp_file(suffix=".pb") as path:
            cmsml.tensorflow.save_graph(path, model, variables_to_constants=True)
            self.assertTrue(os.path.exists(path))

    def test_load_graph(self):
        import google.protobuf as pb

        concrete_func = self.create_tf_function(concrete=True)

        with tmp_file(suffix=".pb") as path_pb, tmp_file(suffix=".pb.txt") as path_txt:
            cmsml.tensorflow.save_graph(path_txt, concrete_func, variables_to_constants=True)
            cmsml.tensorflow.save_graph(path_pb, concrete_func, variables_to_constants=False)

            self.assertTrue(os.path.exists(path_pb))
            self.assertTrue(os.path.exists(path_txt))

            graph = cmsml.tensorflow.load_graph(path_txt)
            self.assertIsInstance(graph, self.tf.Graph)

            graph = cmsml.tensorflow.load_graph(path_pb)
            self.assertIsInstance(graph, self.tf.Graph)

            with self.assertRaises(pb.text_format.ParseError):
                cmsml.tensorflow.load_graph(path_pb, as_text=True)
            with self.assertRaises(pb.message.DecodeError):
                cmsml.tensorflow.load_graph(path_txt, as_text=False)

    def test_load_graph_and_run(self):
        import numpy as np

        tf = self.tf1
        if tf is None:
            return

        _, session = self.create_tf1_graph()
        with tmp_file(suffix=".pb.txt") as path:
            cmsml.tensorflow.save_graph(path, session, variables_to_constants=True,
                output_names=["output"])
            graph = cmsml.tensorflow.load_graph(path)

        session = self.create_tf1_session(graph)
        with graph.as_default():
            x = graph.get_tensor_by_name("input:0")
            y = graph.get_tensor_by_name("output:0")
            out = session.run(y, {x: np.ones((2, 10))})

        self.assertEqual(out.shape, (2, 1))
        self.assertEqual(tuple(out[..., 0]), (1., 1.))

    def test_write_summary(self):
        concrete_func = self.create_tf_function(concrete=True)

        with tmp_dir(create=False) as path:
            cmsml.tensorflow.write_graph_summary(concrete_func.graph, path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(len(os.listdir(path)), 0)

        with tmp_file(suffix=".pb") as graph_path:
            cmsml.tensorflow.save_graph(graph_path, concrete_func)
            with tmp_dir(create=False) as path:
                cmsml.tensorflow.write_graph_summary(graph_path, path)
                self.assertTrue(os.path.exists(path))
                self.assertGreater(len(os.listdir(path)), 0)
                self.assertTrue(os.path.exists(path))
