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
    """Class to create a shared array that can be read/written by parallel
    processes, were data related to the models is going to be stored. For
    memory efficiency reasons, we use RawArrays that will be passed in argument
    to the pool. Each worker will fill a part of the RawArrays. It is
    important that there is no conflict and that two different workers do
    not write on the same section.

    To simplify the interface, from the point of view of the rest of the code,
    this will behave like a regular Numpy array. This is a minimal
    implementation and if new operations are done on these arrays, it may be
    necessary to define them here.
    """
    def __init__(self, size):
        """The RawArray is stored in raw, which is protected by a setter and
        a getter. The array property returns raw as a regular Numpy array. It
        is important to access both the RawArray and the Numpy array forms. The
        conversion from a RawArray to a Numpy array can be costly, in
        particular if it is to just set or get an element. Conversely a
        RawArray is dramatically slower when using slices. To address this
        issue we selectively work with array or raw depending on whether the
        operation is done with a slice or not.
        """
        self.raw = RawArray(ctypes.c_double, size)
        self.size = size
        # By default RawArray initialises all the elements to 0. Setting them to
        # np.nan is preferanble to in case for a reason some elements are never
        # assigned a value during a run
        self.array[:] = np.nan

    def __setitem__(self, idx, data):
        if isinstance(idx, slice):
            self.array[idx] = data
        else:
            self._raw[idx] = data

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.array[idx]
        return self._raw[idx]

    def __len__(self):
        return self.size

    def __rmul__(self, other):
        return other * self.array

    @property
    def array(self):
        return np.ctypeslib.as_array(self._raw)

    @property
    def raw(self):
        return self._raw

    @raw.setter
    def raw(self, raw):
        if isinstance(raw, ctypes.Array):
            self._raw = raw
        else:
            raise TypeError("Type must be RawArray.")
