Command line tools
==================

This document lists and describes the command line tools available in the cmsml package.


.. toctree::
   :maxdepth: 2


cmsml_open_tf_graph
-------------------

.. code-block:: shell

   > cmsml_open_tf_graph --help
   usage: cmsml [-h] [--log-dir LOG_DIR] [--txt] [--binary]
                [--tensorboard-args TENSORBOARD_ARGS]
                graph_path

   Takes a tensorflow graph that was previously saved to a protobuf file and
   opens a tensorboard server to visualize it.

   positional arguments:
     graph_path            the path to the graph to open

   optional arguments:
     -h, --help            show this help message and exit
     --log-dir LOG_DIR, -l LOG_DIR
                           the tensorboard logdir, temporary when not set
     --txt, -t             force reading the graph as text
     --binary, -b          force reading the graph as a binary
     --tensorboard-args TENSORBOARD_ARGS, -a TENSORBOARD_ARGS
                           optional arguments to pass to the tensorboard command
