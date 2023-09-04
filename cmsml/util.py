# coding: utf-8

"""
Helpful functions and utilities.
"""

from __future__ import annotations

__all__ = [
    "is_lazy_iterable", "make_list", "tmp_file", "tmp_dir", "MockModule",
]

import os
import shutil
import tempfile
import contextlib
import importlib
from collections import MappingView
from types import GeneratorType, ModuleType
from typing import Any


lazy_iter_types = (
    GeneratorType,
    MappingView,
    range,
    map,
    enumerate,
)


def is_lazy_iterable(obj: Any) -> bool:
    """
    Returns whether *obj* is iterable lazily, such as generators, range objects, maps, etc.
    """
    return isinstance(obj, lazy_iter_types)


def make_list(obj: Any, cast: bool = True) -> list[Any]:
    """
    Converts an object *obj* to a list and returns it. Objects of types *tuple* and *set* are
    converted if *cast* is *True*. Otherwise, and for all other types, *obj* is put in a new list.
    """
    if isinstance(obj, list):
        return list(obj)
    if is_lazy_iterable(obj):
        return list(obj)
    if isinstance(obj, (tuple, set)) and cast:
        return list(obj)
    return [obj]


def verbose_import(
    module_name: str,
    user: str | None = None,
    package: str | None = None,
    pip_name: str | None = None,
) -> ModuleType:
    try:
        return importlib.import_module(module_name, package=package)
    except ImportError as e:
        if user:
            e.msg += f" but is required by {user}"
        if pip_name:
            e.msg += f" (you may want to try 'pip install --user {pip_name}')"
        raise e


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


class MockModule(object):
    """
    Mockup object that resembles a module with arbitrarily deep structure such that, e.g.,

    .. code-block:: python

        tf = MockModule("tensorflow")
        print(tf.Graph)
        # -> "<MockupModule 'tf' at 0x981jald1>"

    will always succeed at declaration.

    .. py:attribute:: _name
       type: str

       The name of the mock module.
    """

    def __init__(self, name: str):
        super().__init__()

        self._name = name

    def __getattr__(self, attr: str) -> "MockModule":
        return type(self)(f"{self._name}.{attr}")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} '{self._name}' at {hex(id(self))}>"

    def __call__(self, *args, **kwargs) -> None:
        raise Exception(f"{self._name} is a mock module and cannot be called")

    def __nonzero__(self) -> bool:
        return False

    def __bool__(self) -> bool:
        return False

    def __or__(self, other) -> Any:
        # forward union type hints
        return type(self) | other
