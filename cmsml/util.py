# coding: utf-8

"""
Helpful functions and utilities.
"""

from __future__ import annotations

__all__ = [
    "is_lazy_iterable", "make_list", "tmp_file", "tmp_dir", "MockModule",
]

import os
import time
import shutil
import tempfile
import contextlib
import subprocess
import signal
import importlib
import six

from collections.abc import MappingView
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


class MockModule(object):
    """
    Mockup object that resembles a module with arbitrarily deep structure such that, e.g.,

    .. code-block:: python

        tf = MockModule("tensorflow")
        print(tf.Graph)
        # -> "<MockupModule 'tf' at 0x981jald1>"

    will always succeed at declaration time.

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
