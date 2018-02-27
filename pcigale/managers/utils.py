# -*- coding: utf-8 -*-
# Copyright (C) 2018 Universidad de Antofagasta
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Héctor Salas

"""
Various utility functions for pcigale manager modules
"""

import ctypes
import numpy as np

from multiprocessing.sharedctypes import RawArray
from ..warehouse import SedWarehouse

def get_info(cls):
    warehouse = SedWarehouse()
    sed = warehouse.get_sed(cls.conf['sed_modules'],
                            cls.params.from_index(0))
    info = list(sed.info.keys())
    info.sort()

    return (info, sed.mass_proportional_info)

class SharedArray(object):
    """Class to Create a shared array that can be read/written by parallel
    processes, were data related to the models is going to be stored. For
    memory efficiency reasons, we use RawArrays that will be passed in argument
    to the pool. Each worker will fill a part of the RawArrays. It is
    important that there is no conflict and that two different workers do
    not write on the same section.

    """
    def __init__(self, shape):
        self._shape = shape
        self._data = RawArray(ctypes.c_double, int(np.product(self._shape)))

    @property
    def data(self):
        return np.ctypeslib.as_array(self._data).reshape(self._shape)

    @property
    def shape(self):
        return self._shape
