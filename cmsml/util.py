# coding: utf-8

"""
Helpful functions and utilities.
"""

__all__ = [
    "is_lazy_iterable", "make_list", "tmp_file", "tmp_dir",
]


import os
import shutil
import tempfile
import contextlib
import types
import collections
import importlib

import six


def is_lazy_iterable(obj):
    """
    Returns whether *obj* is iterable lazily, such as generators, range objects, maps, etc.
    """
    iter_types = (
        types.GeneratorType, collections.MappingView, six.moves.range, six.moves.map, enumerate,
    )
    return isinstance(obj, iter_types)


def make_list(obj, cast=True):
    """
    Converts an object *obj* to a list and returns it. Objects of types *tuple* and *set* are
    converted if *cast* is *True*. Otherwise, and for all other types, *obj* is put in a new list.
    """
    if isinstance(obj, list):
        return list(obj)
    elif is_lazy_iterable(obj):
        return list(obj)
    elif isinstance(obj, (tuple, set)) and cast:
        return list(obj)
    else:
        return [obj]


def verbose_import(user, module_name, package=None, pip_name=None):
    try:
        return importlib.import_module(module_name, package=package)
    except ImportError as e:
        e.msg += " but is required by {}".format(user)
        if pip_name:
            e.msg += " (you may want to try 'pip install --user {}')".format(pip_name)
        raise


@contextlib.contextmanager
def tmp_file(create=False, delete=True, **kwargs):
    """
    Prepares a temporary file and opens a context yielding its path. When *create* is *True*, the
    file is created before the context is opened, and deleted upon closing if *delete* is *True*.
    All *kwargs* are forwarded to :py:func:`tempfile.mkstemp`.
    """
    path = tempfile.mkstemp(**kwargs)[1]

    exists = os.path.exists(path)
    if not create and exists:
        os.remove(path)
    elif create and not exists:
        open(path, "a").close()

    try:
        yield path
    finally:
        if delete and os.path.exists(path):
            os.remove(path)


@contextlib.contextmanager
def tmp_dir(create=True, delete=True, **kwargs):
    """
    Prepares a temporary directory and opens a context yielding its path. When *create* is *True*,
    the directory is created before the context is opened, and deleted upon closing if *delete* is
    *True*. All *kwargs* are forwarded to :py:func:`tempfile.mkdtemp`.
    """
    path = tempfile.mkdtemp(**kwargs)

    exists = os.path.exists(path)
    if not create and exists:
        shutil.rmtree(path)
    elif create and not exists:
        os.makedirs(path)

    try:
        yield path
    finally:
        if delete and os.path.exists(path):
            shutil.rmtree(path)
