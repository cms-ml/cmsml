# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cmsml
from cmsml.util import tmp_dir

from . import CMSMLTestCase


class TfCompileTestCase(CMSMLTestCase):
    def __init__(self, *args, **kwargs):
        super(TfCompileTestCase, self).__init__(*args, **kwargs)

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
        from cmsml.scripts.compile_tf_graph import compile_tf_graph

        # check only preparation process for aot, but do not aot compile
        tf = self.tf

        model = self.create_test_model(tf)

        with tmp_dir(create=False) as model_path:
            tf.saved_model.save(model, model_path)

            with tmp_dir(create=False) as static_saved_model_path:
                batch_sizes = [1, 2]

                compile_tf_graph(model_path=model_path,
                                output_path=static_saved_model_path,
                                batch_sizes=batch_sizes,
                                input_serving_key=tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
                                output_serving_key=None,
                                compile_prefix=None,
                                compile_class=None)

                # load model and check input shape
                loaded_static_model = cmsml.tensorflow.load_model(static_saved_model_path)
                for batch_size in batch_sizes:
                    # first entry is empty, second contains inputs tuple(tensorspecs)
                    model_static_inputs = loaded_static_model.signatures[f"serving_default__{batch_size}"].structured_input_signature[1]  # noqa

                    expected_model_static_inputs = {
                        f"first__bs{batch_size}": tf.TensorSpec(
                            shape=(batch_size, 2),
                            dtype=tf.float32,
                            name=f"first__bs{batch_size}",
                        ),
                        f"second__bs{batch_size}": tf.TensorSpec(
                            shape=(batch_size, 3),
                            dtype=tf.float32,
                            name=f"second__bs{batch_size}",
                        ),
                        f"third__bs{batch_size}": tf.TensorSpec(
                            shape=(batch_size, 10),
                            dtype=tf.float32,
                            name=f"third__bs{batch_size}",
                        ),
                    }

                    self.assertDictEqual(model_static_inputs, expected_model_static_inputs)

                # throw error if compilation happens with illegal batch size
                with self.assertRaises(ValueError):
                    compile_tf_graph(model_path=model_path,
                                output_path=static_saved_model_path,
                                batch_sizes=[-1,],
                                input_serving_key=tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
                                output_serving_key=None,
                                compile_prefix=None,
                                compile_class=None)

    def test_compile_tf_graph_static_aot_compilation(self):
        from cmsml.scripts.compile_tf_graph import compile_tf_graph

        # check aot compilation
        tf = self.tf
        model = self.create_test_model(tf)

        with tmp_dir(create=False) as model_path:
            tf.saved_model.save(model, model_path)

            with tmp_dir(create=False) as static_saved_model_path:
                batch_sizes = [1, 2]
                compile_tf_graph(model_path=model_path,
                                output_path=static_saved_model_path,
                                batch_sizes=batch_sizes,
                                input_serving_key=tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
                                output_serving_key=None,
                                compile_prefix="aot_model_bs_{}",
                                compile_class="bs_{}")

                aot_dir = os.path.join(static_saved_model_path, "aot")
                for batch_size in batch_sizes:
                    aot_model_header = os.path.join(aot_dir, "aot_model_bs_{}.h".format(batch_size))
                    aot_model_object = os.path.join(aot_dir, "aot_model_bs_{}.o".format(batch_size))

                    self.assertTrue(os.path.exists(aot_model_object))
                    self.assertTrue(os.path.exists(aot_model_header))
