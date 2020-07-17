# coding: utf-8

from __future__ import absolute_import

"""
Helpful functions and utilities.
"""

__all__ = ["import_tf"]


def import_tf():
    """
    Imports TensorFlow and returns a 3-tuple containing the module itself, the v1 compatibility
    API (i.e. the TensorFlow module itself if v1 is the primarily installed version), and the
    package version string. Example:

    .. code-block:: python

        tf, tf1, tf_version = import_tf()

    At some point in the future, when v1 support might get fully removed from TensorFlow 2 or
    higher, the second tuple element might be *None*.
    """
    import tensorflow as tf

    # keep a reference to the v1 API as long as v2 provides compatibility
    tf1 = None
    tf_version = tf.__version__.split(".", 2)
    if tf_version[0] == "1":
        tf1 = tf
    elif getattr(tf, "compat", None) is not None and getattr(tf.compat, "v1", None) is not None:
        tf1 = tf.compat.v1

    return tf, tf1, tf_version
