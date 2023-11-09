# coding: utf-8

"""
Lazy loading of cmsml submodules to update the global cmsml dict when a module is accessed the first
time.

We want to avoid the need for manually importing certain cmsml submdoules on the user side, such as

.. code-block:: python

    import cmsml
    import cmsml.tensorflow
    import cmsml.keras
    import cmsml....

A common approach is to import and thereby provide all submodules already when the main cmsml module
is loaded, but this bears the risk of external packages being used in these submodules that are not
available in the user environment. The lazy loading mechanism thus imports the accessed submodule
on first access.
"""

from __future__ import annotations

__all__ = []

import importlib
from typing import Any

import cmsml


_lazy_modules = None


def started() -> bool:
    return _lazy_modules is not None


def start(module_names: list[str]) -> None:
    global _lazy_modules

    # make sure to only start once
    if started():
        return

    # create placeholders which hook into the cmsml module dict and store names
    _lazy_modules = []
    for module_name in module_names:
        ModulePlaceholder(module_name)
        _lazy_modules.append(module_name)


class ModulePlaceholder(object):

    def __init__(self, module_name: str):
        super().__init__()

        self.module_name = module_name

        # add to the cmsml module dict
        cmsml.__dict__.setdefault(self.module_name, self)

    def __getattr__(self, attr: str) -> Any:
        # when this placeholder is still in the cmsml module dict,
        # import the actual module and replace it
        if cmsml.__dict__.get(self.module_name) in (None, self):
            module = importlib.import_module(f"cmsml.{self.module_name}")
            cmsml.__dict__[self.module_name] = module

        # return the proper attribute which is most likely only required for the first call
        return getattr(cmsml.__dict__[self.module_name], attr)
