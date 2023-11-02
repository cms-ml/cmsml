import cmsml

from . import CMSMLTestCase

import os

from cmsml.util import tmp_file, tmp_dir
from cmsml.scripts.compile_tf_graph import compile_tf_graph


class TfCompileTestCase(CMSMLTestCase):
    def __init__(self, *args, **kwargs):
        super(TfCompileTestCase, self).__init__(*args, **kwargs)

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


    def create_keras_model(self, tf):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.InputLayer(input_shape=(10,), dtype=tf.float32, name="input"))
        model.add(tf.keras.layers.BatchNormalization(axis=1, renorm=True))
        model.add(tf.keras.layers.Dense(100, activation="tanh"))
        model.add(tf.keras.layers.BatchNormalization(axis=1, renorm=True))
        model.add(tf.keras.layers.Dense(3, activation="softmax", name="output"))
        return model

    def create_test_model(self, tf):
        # model with 2 input nodes and 1 output node, with non-static batchsize
        x1 = tf.keras.Input(shape=(2,), name="first")
        x2 = tf.keras.Input(shape=(3,), name="second")
        x3 = tf.keras.Input(shape=(10,), name="third")

        x = tf.concat([x1, x2], axis=1)
        a1 = tf.keras.layers.Dense(10, activation="elu")(x)
        y = tf.keras.layers.Dense(5, activation="softmax")(a1)
        model = tf.keras.Model(inputs=(x1, x2, x3), outputs=y)
        return model


    def test_compile_tf_graph_static_preparation(self):
        # check only preparation process for aot, but do not aot compile
        model = self.create_test_model(self, self.tf)

        with tmp_dir(create=False) as model_path:
            self.tf.saved_model.save(model, model_path)

            with tmp_dir(create=False) as static_saved_model_path:
                compile_tf_graph(model_path=model_path,
                                output_path=static_saved_model_path,
                                batch_sizes=[1, 2],
                                input_serving_key=self.tf2.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
                                output_serving_key=None,
                                compile_prefix=None,
                                compile_class=None)

                # check if saving worked correctly
                self.assertTrue(os.path.exists(static_saved_model_path))

                # load model and check input shape
                loaded_static_model = cmsml.load_model(static_saved_model_path)

                # first entry is empty, second contains inputs tensorspecs
                model_static_inputs = loaded_static_model.signatures['serving_default__2'].structured_input_signature[1],

                expected_model_static_inputs = {"first__bs2": self.tf.TensorSpec(shape=(2, 2), dtype=self.tf.float32, name="first__bs2"),
                                         "second_bs2": self.tf.TensorSpec(shape=(2, 3), dtype=self.tf.float32, name="second__bs2"),
                                         "third_bs2": self.tf.TensorSpec(shape=(2, 10), dtype=self.tf.float32, name="third__bs2")}

                self.assertDictEqual(model_static_inputs, expected_model_static_inputs)

    def test_compile_tf_graph_static_aot_compilation(self):
        # check only preparation process for aot, but do not aot compile
        model = self.create_test_model(self, self.tf)

        with tmp_dir(create=False) as model_path:
            self.tf.saved_model.save(model, model_path)

            with tmp_dir(create=False) as static_saved_model_path:
                batch_sizes = [1, 2]
                compile_tf_graph(model_path=model_path,
                                output_path=static_saved_model_path,
                                batch_sizes=batch_sizes,
                                input_serving_key=self.tf2.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
                                output_serving_key=None,
                                compile_prefix='aot_model_bs_{}',
                                compile_class='bs_{}')

                aot_path = os.path.join(static_saved_model_path,"aot")

                for batch_size in batch_sizes:
                    aot_model_header = 'aot_model_bs_{}.h'.format(batch_size)
                    aot_model_object = 'aot_model_bs_{}.o'.format(batch_size)

                    self.assertTrue(os.path.exits(os.path.join(static_saved_model_path, aot_model_object)))
                    self.assertTrue(os.path.exits(os.path.join(static_saved_model_path, aot_model_header)))
