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
        ORDER_OF_INPUT_NODES = ["first", "second", "third"]
        # global paths to the temp dirs containing the test models
        cls.paths = TestConvertModel.save_test_model()
        cls.keras_converter = ModelConverter(
            src_dir=cls.paths["keras"], batch_sizes=BATCHSIZES, inputs=ORDER_OF_INPUT_NODES)
        cls.tf_converter = ModelConverter(
            src_dir=cls.paths["tf"], batch_sizes=BATCHSIZES, inputs=ORDER_OF_INPUT_NODES)

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

        self.assertEqual(result_bs, ModelConverter._map_batch_sizes(str_bs))
        self.assertEqual(result_bs, ModelConverter._map_batch_sizes(tuple_bs))
        self.assertEqual(result_bs, ModelConverter._map_batch_sizes(list(tuple_bs)))

        with self.assertRaises(Exception):
            ModelConverter._map_batch_sizes(fail_bs)

    def test__is_keras(self):
        self.assertTrue(ModelConverter._is_keras(Path(self.paths["keras"])))
        self.assertFalse(ModelConverter._is_keras(Path(self.paths["tf"])))

    def test__static_tensorspecs(self):
        batch_size = self.tf_converter.batch_sizes[0]
        tf_static_spec = set(self.tf_converter._static_tensorspecs(batch_size))
        keras_static_spec = set(self.tf_converter._static_tensorspecs(batch_size))

        # both models should have result in the same TensorSpecs after the conversion
        self.assertEqual(tf_static_spec, keras_static_spec)

        # models should result in this list with TensorSpecs
        should_be_static_spec = {tf.TensorSpec(shape=(1, 2), dtype=tf.float32, name='first'),
                                 tf.TensorSpec(shape=(1, 3), dtype=tf.float32, name='second'),
                                 tf.TensorSpec(shape=(1, 10), dtype=tf.float32, name='third')}

        self.assertEqual(tf_static_spec, should_be_static_spec)

    def test_inputs(self):

        dummy_inputs = [tf.TensorSpec(shape=(None, 2), dtype=tf.float32, name='first'),
                        tf.TensorSpec(shape=(None, 3), dtype=tf.float32, name='second'),
                        tf.TensorSpec(shape=(None, 10), dtype=tf.float32, name='third')]

        keras_inputs = self.keras_converter.inputs
        tf_inputs = self.tf_converter.inputs

        # inputs of the concrete function should be in the same order as dummy_inputs
        self.assertEqual(dummy_inputs, keras_inputs)
        self.assertEqual(dummy_inputs, tf_inputs)


    def test__static_concrete_function(self):
        batchsize = 1

        # hold = {}

        # for i in range(0, 20):
        #     m= ModelConverter(
        #         src_dir=self.paths["tf"], batch_sizes="1 2 4", inputs=["first", "second", "third"])

        #     hold[i] = {}
        #     hold[i]['names'] = m.graph._arg_keywords
        #     hold[i]['structured'] = m.graph.structured_input_signature
        #     hold[i]['inputs'] = m.graph.inputs
        #     hold[i]['concrete'] = m.graph

        from IPython import embed
        embed()

        keras_concrete = self.keras_converter._static_concrete_function(batchsize)

        tf_concrete = self.tf_converter._static_concrete_function(batchsize)


        # both concrete functions should have the same input_TensorSpec
        self.assertEqual(keras_concrete.structured_input_signature,
                        tf_concrete.structured_input_signature)

        # # concrete functions can be called and should result in the same result as the model
        # a = tf.constant(((1., 2.),))
        # b = tf.constant(((1., 2., 3.),))

        # keras_concrete_result = keras_concrete(a, b)
        # tf_concrete_result = tf_concrete(a, b)
        # keras_model_result = self.keras_converter.model(a, b)
        # tf_model_result = self.tf_converter.model(a, b)

        # self.assertEqual(keras_concrete_result, keras_model_result)
        # self.assertEqual(tf_concrete_result, tf_model_result)
        # self.assertEqual(keras_concrete_result, tf_concrete_result)

        # from IPython import embed
        # embed()


if __name__ == '__main__':
    unittest.main()
