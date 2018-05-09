# -*- coding: utf-8 -*-
# Copyright (C) 2012, 2013 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly

from ..sed import SED
from .. import sed_modules


class SedWarehouse(object):
    """Create, cache and store SED

    This object is responsible for creating SED and storing them in a memory
    cache or a database.
    """

    def __init__(self, nocache=None):
        """Instantiate an SED warehouse

        Parameters
        ----------
        nocache: list
            SED module or list of the SED modules that are not to be cached,
        trading CPU for memory.

        """
        if nocache is None:
            self.nocache = []
        elif isinstance(nocache, list) is True:
            self.nocache = nocache
        elif isinstance(nocache, str) is True:
            self.nocache = [nocache]
        else:
            raise TypeError("The nocache argument must be a list or an str.")

        self.sed_cache = {}
        self.module_cache = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_module_cached(self, name, **kwargs):
        """Get the SED module using the internal cache.

        Parameters
        ----------
        name: string
            Module name.

        The other keyword parameters are the module parameters.

        Returns
        -------
        a pcigale.sed_modules.Module instance

        """
        if name in self.nocache:
            module = sed_modules.get_module(name, **kwargs)
        else:
            # Use the name of the module and the values of the parameters as a
            # key to the cache. This works because the parameters are stored in
            # ordered dictionaries.
            module_key = (name, tuple(kwargs.values()))
            if module_key in self.module_cache:
                module = self.module_cache[module_key]
            else:
                module = sed_modules.get_module(name, **kwargs)
                self.module_cache[module_key] = module

        return module

    def partial_clear_cache(self, n_modules_max):
        """Clear the cache of SEDs that are not relevant anymore

        To do partial clearing of the cache, we go through the entire cache
        and delete the SEDs that have more than a given number of modules.
        This is done by computing the index of the module that has a changed
        parameter. This means that SEDs with this number of modules or more
        are not needed anymore to compute new models and we can discard them.
        Passing 0 as an argument empties the cache completely.

        Parameters
        ----------
        n_modules_max: int
            Maximum number of modules. All SED with at least this number of
            modules have to be discarded

        """
        if n_modules_max > -1:
            for k in list(self.sed_cache.keys()):
                if len(k) > n_modules_max:
                    del self.sed_cache[k]

    def get_sed(self, module_list, parameter_list):
        """Get the SED corresponding to the module and parameter lists

        If the SED was cached, get it from the cache. If it is not, create it
        and add it the the cache. The method is recursive to permit caching
        partial SED.

        Parameters
        ----------
        module_list: iterable
            List of module names in the order they have to be used to
            create the SED.
        parameter_list: iterable
            List of the parameter dictionaries corresponding to each
            module of the module_list list.

        Returns
        -------
        sed: pcigale.sed
            The SED made from the given modules with the given parameters.

        """
        module_list = list(module_list)
        parameter_list = list(parameter_list)

        # Use the values of the parameters of all the modules as the key for
        # the cache. This works because the parameters are stored in ordered
        # dictionaries.
        key = tuple(tuple(par.values()) for par in parameter_list)

        sed = self.sed_cache.get(key)
        if sed is None:
            mod = self.get_module_cached(module_list.pop(),
                                         **parameter_list.pop())

            if (len(module_list) == 0):
                sed = SED()
            else:
                sed = self.get_sed(module_list, parameter_list).copy()

            mod.process(sed)
            self.sed_cache[key] = sed

        return sed
