import sys
sys.path.append("..")
sys.path.append("../..")

import tensorflow as tf
from convert_model import ModelConverter
from util import tmp_dir, tmp_file
from pathlib import Path
import shutil
import unittest


class TestConvertModel(unittest.TestCase):

    def setUp(self):
        pass

    @classmethod
    def setUpClass(cls):
        BATCHSIZES = "1 2 4"
        # global paths to the temp dirs containing the test models
        cls.paths = TestConvertModel.save_test_model()
        cls.keras_converter = ModelConverter(
            src_dir=cls.paths["keras"], batch_sizes=BATCHSIZES)
        cls.tf_converter = ModelConverter(
            src_dir=cls.paths["tf"], batch_sizes=BATCHSIZES)

    @classmethod
    def tearDownClass(cls):
        # remove all dirs in tmp after finish test
        for path in cls.paths.values():
            shutil.rmtree(path)

    @staticmethod
    def create_test_model():
        # model with 2 input nodes and 1 output node, with non-static batchsize
        x1 = tf.keras.Input(shape=(2,), name="first")
        x2 = tf.keras.Input(shape=(3,), name="second")
        x3 = tf.keras.Input(shape=(10,), name="third")

        x = tf.concat([x1, x2], axis=1)
        a1 = tf.keras.layers.Dense(10, activation="elu")(x)
        y = tf.keras.layers.Dense(5, activation="softmax")(a1)
        model = tf.keras.Model(inputs=(x1, x2, x3), outputs=y)
        return model

    @staticmethod
    def save_test_model():
        model = TestConvertModel.create_test_model()
        with tmp_dir(create=False, delete=False) as path_tf:
            # save model in tmp as keras or tf2 saved_model
            tf.saved_model.save(model, path_tf)
        with tmp_dir(create=False, delete=False) as path_keras:
            model.save(path_keras, overwrite=True, include_optimizer=False)
        paths = {"keras": path_keras, "tf": path_tf}
        return paths

    def test__map_batch_sizes(self):
        # maps input str of batch sizes or list tuple into
        # a series of ints to be used by the function
        result_bs = (1, 2, 4)

        str_bs = "1 2 4"
        tuple_bs = (1., 2., 4.)
        fail_bs = "1;2;4"
        fail_bs2 = "1 2  4"

        self.assertEqual(result_bs, ModelConverter._map_batch_sizes(str_bs))
        self.assertEqual(result_bs, ModelConverter._map_batch_sizes(tuple_bs))
        self.assertEqual(result_bs, ModelConverter._map_batch_sizes(list(tuple_bs)))

        for will_fail_bs in (fail_bs, fail_bs2):
            with self.assertRaises(Exception):
                ModelConverter._map_batch_sizes(will_fail_bs)

    def test__is_keras(self):
        self.assertTrue(ModelConverter._is_keras(Path(self.paths["keras"])))
        self.assertFalse(ModelConverter._is_keras(Path(self.paths["tf"])))

    def test__static_tensorspecs(self):
        # should create a tensorspec with just the SHAPE being static
        batch_size = self.tf_converter.batch_sizes[0]
        tf_static_spec = self.tf_converter._static_tensorspecs(batch_size)
        keras_static_spec = self.keras_converter._static_tensorspecs(batch_size)

        # both models should have result in the same TensorSpecs after the conversion
        self.assertEqual(tf_static_spec, keras_static_spec)

        # models tensorspec should result this dict with TensorSpecs
        # thus, NAME, DTYPE should be preserve
        should_be_static_spec = {"first": tf.TensorSpec(shape=(1, 2), dtype=tf.float32, name="first"),
                                 "second": tf.TensorSpec(shape=(1, 3), dtype=tf.float32, name="second"),
                                 "third": tf.TensorSpec(shape=(1, 10), dtype=tf.float32, name="third")}

        self.assertEqual(tf_static_spec, should_be_static_spec)

    def test__create_static_signatures(self):
        # concrete functions between keras and tensorflow
        # should have the same inputs/outputs with static batch size

        keras_concretes = self.keras_converter.create_static_signatures()
        tf_concretes = self.tf_converter.create_static_signatures()

        # both concrete functions should result in same structured signature
        equality_check = all([keras_concrete.structured_input_signature == tf_concrete.structured_input_signature for keras_concrete,
                             tf_concrete in zip(keras_concretes.values(), tf_concretes.values())])
        self.assertTrue(equality_check)

        # both signatures should result in same prediction

        # concrete functions can be called and should predict the same as the model
        inputs = {
            "first": tf.ones(shape=(1, 2)),
            "second": tf.ones(shape=(1, 3)),
            "third": tf.ones(shape=(1, 10)),
        }

        with tf.device("/CPU:*"):
            keras_result = keras_concretes["batch_size_1"](**inputs)
            tf_result = tf_concretes["batch_size_1"](**inputs)
            self.assertTrue(tf.math.reduce_all(tf.math.equal(
                keras_result["dense_1"], tf_result["dense_1"])).numpy())


if __name__ == "__main__":
    unittest.main()
