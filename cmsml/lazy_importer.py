# coding: utf-8

"""
Lazy importer that updates the cmsml module dict to prevent
unavailable modules from being loaded.
"""

__all__ = ["start"]


import importlib

import cmsml


_started = False


def start(module_names):
    global _started

    # make sure to only start once
    if _started:
        return
    _started = True

    # create placeholder which hook into the cmsml module dict
    for module_name in module_names:
        ModulePlaceholder(module_name)


class ModulePlaceholder(object):

    def __init__(self, module_name):
        super(ModulePlaceholder, self).__init__()

        self.module_name = module_name

        # add to the cmsml module dict
        cmsml.__dict__[self.module_name] = self

    def __getattr__(self, attr):
        # when this placeholder is still in the cmsml module dict,
        # import the actual module and replace it
        if cmsml.__dict__.get(self.module_name) in (None, self):
            module = importlib.import_module("cmsml." + self.module_name)
            cmsml.__dict__[self.module_name] = module

        # return the proper attribute which is most likely only required for the first getattr call
        return getattr(cmsml.__dict__[self.module_name], attr)
