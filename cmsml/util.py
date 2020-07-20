# coding: utf-8

"""
Helpful functions and utilities.
"""

__all__ = ["tmp_file", "tmp_dir"]


import os
import shutil
import tempfile
import contextlib


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
