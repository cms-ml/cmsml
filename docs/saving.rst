Saving and converting models
============================

This document describes how to save trained models in various ways and how to convert them to formats used by other frameworks.

Relevant API docs:

- Writing TensorFlow protobuf files: :py:func:`cmsml.tensorflow.save_graph`


.. toctree::
   :maxdepth: 2


.. _graph_to_file:

``tf.Graph`` to protobuf file (TF 1)
------------------------------------

.. code-block:: python

   import tensorflow as tf  # assuming this is tf 1
   import cmsml

   # define graph and session
   graph = tf.Graph()
   with graph.as_default():
       x = tf.placeholder(tf.float32, [None, 10], name="input")
       ...
       y = tf.tanh(..., name="output")  # noqa
   session = tf.Session(graph=graph)

   # convert to binary protobuf
   cmsml.tensorflow.save_graph("/path/to/graph.pb", graph)

   # convert to human readable protobuf
   cmsml.tensorflow.save_graph("/path/to/graph.pb.txt", graph)

   # convert to binary protobuf, convert variables to constants (recommended for use in CMSSW)
   # note that this requires the session to be passed as well as the names of the output operations
   cmsml.tensorflow.save_graph("/path/to/graph.pb", session, variables_to_constants=True,
       output_names=["output"])


``tf.function`` to protobuf file (TF 2)
---------------------------------------

For deeper insights into ``tf.function``, the concepts of signature tracing, polymorphic and concrete functions, see the guide on `Better performance with tf.function <https://www.tensorflow.org/guide/function>`__.

Defining the function
+++++++++++++++++++++

The standard way of defining a ``tf.function`` is probably:

.. code-block:: python

   import tensorflow as tf  # assuming this is tf 2
   import cmsml

   # define tf function
   @tf.function
   def model(x):
       ...
       return tf.tanh(...)

The ``model`` function will accept various different types for the input argument ``x``. In TensorFlow terms it is therefore called *polymorphic*. Every invocation with a differently shaped or typed tensor, or even Python object will trigger a new graph to be saved internally. This mechanism is referred to as signature tracing. By default, polymorphic functions have no internal graph representation and thus cannot be saved directly. This is underlined by the fact that the function body is never even evaluated when ``model`` is not called with a certain value for ``x``.


Saving the graph
++++++++++++++++

There are two ways to obtain a function that can be saved:

**1. Define an input signature**

.. code-block:: python

   @tf.function(input_signature=(tf.TensorSpec(shape=[2, 10], dtype=tf.float32),))
   def model(x):
       ...

When the ``input_signature`` is a priory known (or *frozen*), TensorFlow can interpolate types and shapes of operations and tensors within the function body, and you can save the internal graph.

.. code-block:: python

   # convert to binary protobuf
   cmsml.tensorflow.save_graph("/path/to/graph.pb", model)

   # convert to human readable protobuf
   cmsml.tensorflow.save_graph("/path/to/graph.pb.txt", model)

   # convert to binary protobuf, convert variables to constants (recommended for use in CMSSW)
   cmsml.tensorflow.save_graph("/path/to/graph.pb", model, variables_to_constants=True)


**2. Create a concrete function**

Concrete functions are the results of signature tracing and are usually held internally. However, you can create a concrete function from a polymorphic function via:

.. code-block:: python

   concrete_model = model.get_concrete_function(tf.TensorSpec(shape=[2, 10], dtype=tf.float32))


Saving a concrete function is identical to saving a frozen one as shown above:

.. code-block:: python

   # convert to binary protobuf
   cmsml.tensorflow.save_graph("/path/to/graph.pb", concrete_model)



Keras model to protobuf file (TF 1)
-----------------------------------

When working with keras and the TensorFlow 1 backend, saving the model to a protobuf graph file works the same way as saving a ``tf.Graph`` as :ref:`described above<graph_to_file>`. Just make sure that the model is either compiled, or that an ``input_shape`` is passed to the first layer:

.. code-block:: python

   import tensorflow as tf  # assuming this is tf 1
   import cmsml

   model = tf.keras.Sequential()
   model.add(tf.keras.layers.InputLayer(input_shape=(10,), dtype=tf.float32, name="input"))
   model.add(tf.keras.layers.Dense(100, activation="tanh"))
   model.add(tf.keras.layers.Dense(3, activation="softmax", name="output"))

   # convert graph from internal session to binary protobuf
   cmsml.tensorflow.save_graph("/path/to/graph.pb", tf.keras.backend.get_session())


Keras model to protobuf file (TF 2)
-----------------------------------

When you build your model with keras and TensorFlow 2, you just need to pass it to :py:func:`~cmsml.tensorflow.save_graph`. As above, just make sure the model is either compiled, or that an ``input_shape`` is passed to the first layer:

.. code-block:: python

   import tensorflow as tf  # assuming this is tf 2
   import cmsml

   model = tf.keras.Sequential()
   model.add(tf.keras.layers.InputLayer(input_shape=(10,), dtype=tf.float32, name="input"))
   model.add(tf.keras.layers.Dense(100, activation="tanh"))
   model.add(tf.keras.layers.Dense(3, activation="softmax", name="output"))

   # convert graph from internal session to binary protobuf
   cmsml.tensorflow.save_graph("/path/to/graph.pb", model)
