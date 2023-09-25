# coding: utf-8

"""
Helpful functions and utilities.
"""

__all__ = [
    "is_lazy_iterable", "make_list", "tmp_file", "tmp_dir", "colored",
]


import os
import sys
import time
import shutil
import tempfile
import contextlib
import subprocess
import signal
import types
import importlib

import six


lazy_iter_types = (
    types.GeneratorType,
    six.moves.collections_abc.MappingView,
    six.moves.range,
    six.moves.map,
    enumerate,
)


def is_lazy_iterable(obj):
    """
    Returns whether *obj* is iterable lazily, such as generators, range objects, maps, etc.
    """
    return isinstance(obj, lazy_iter_types)


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


def verbose_import(module_name, user=None, package=None, pip_name=None):
    try:
        return importlib.import_module(module_name, package=package)
    except ImportError:
        e_type, e, traceback = sys.exc_info()
        msg = str(e)
        if user:
            msg += " but is required by {}".format(user)
        if pip_name:
            msg += " (you may want to try 'pip install --user {}')".format(pip_name)
        six.reraise(e_type, e_type(msg), traceback)


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


_shell_colors = {
    "default": 39,
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
}


def colored(s: str, color: str = "white") -> str:
    """
    Returns a string *s* in a shell-colored representation.
    """
    color_id = _shell_colors.get(color, 39)
    return "\033[{}m{}\033[0m".format(color_id, s)


def interruptable_popen(*args, **kwargs):
    """ interruptable_popen(*args, stdin_callback=None, stdin_delay=0, interrupt_callback=None, kill_timeout=None, **kwargs)  # noqa
    Shorthand to :py:class:`Popen` followed by :py:meth:`Popen.communicate` which can be interrupted
    by *KeyboardInterrupt*. The return code, standard output and standard error are returned in a
    3-tuple.

    *stdin_callback* can be a function accepting no arguments and whose return value is passed to
    ``communicate`` after a delay of *stdin_delay* to feed data input to the subprocess.

    *interrupt_callback* can be a function, accepting the process instance as an argument, that is
    called immediately after a *KeyboardInterrupt* occurs. After that, a SIGTERM signal is send to
    the subprocess to allow it to gracefully shutdown.

    When *kill_timeout* is set, and the process is still alive after that period (in seconds), a
    SIGKILL signal is sent to force the process termination.

    All other *args* and *kwargs* are forwarded to the :py:class:`Popen` constructor.
    """
    # get kwargs not being passed to Popen
    stdin_callback = kwargs.pop("stdin_callback", None)
    stdin_delay = kwargs.pop("stdin_delay", 0)
    interrupt_callback = kwargs.pop("interrupt_callback", None)
    kill_timeout = kwargs.pop("kill_timeout", None)

    # start the subprocess in a new process group
    kwargs["preexec_fn"] = os.setsid
    p = subprocess.Popen(*args, **kwargs)

    # get stdin
    stdin_data = None
    if callable(stdin_callback):
        if stdin_delay > 0:
            time.sleep(stdin_delay)
        stdin_data = stdin_callback()
        if isinstance(stdin_data, six.string_types):
            stdin_data = (stdin_data + "\n").encode("utf-8")

    # handle interrupts
    try:
        out, err = p.communicate(stdin_data)
    except KeyboardInterrupt:
        # allow the interrupt_callback to perform a custom process termination
        if callable(interrupt_callback):
            interrupt_callback(p)

        # when the process is still alive, send SIGTERM to gracefully terminate it
        pgid = os.getpgid(p.pid)
        if p.poll() is None:
            os.killpg(pgid, signal.SIGTERM)

        # when a kill_timeout is set, and the process is still running after that period,
        # send SIGKILL to force its termination
        if kill_timeout is not None:
            target_time = time.perf_counter() + kill_timeout
            while target_time > time.perf_counter():
                time.sleep(0.05)
                if p.poll() is not None:
                    # the process terminated, exit the loop
                    break
            else:
                # check the status again to avoid race conditions
                if p.poll() is None:
                    os.killpg(pgid, signal.SIGKILL)

        # transparently reraise
        raise

    # decode outputs
    if out is not None:
        out = out.decode("utf-8")
    if err is not None:
        err = err.decode("utf-8")

    return p.returncode, out, err
