# -*- coding: utf-8 -*-
# Copyright (C) 2017 Universidad de Antofagasta
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Médéric Boquien

"""This class manages the handling of models in the code. It contains the
fluxes and the physical properties and all the necessary information to
compute them, such as the configuration, the observations, and the parameters
of the models.
"""

import ctypes
from multiprocessing.sharedctypes import RawArray

from astropy.table import Table, Column
import numpy as np

from ..warehouse import SedWarehouse


class ModelsManager(object):
    """A ModelsManager contains the fluxes and the properties of all the
    models. In addition it contains all the information to understand how the
    models have been computed, what block of the grid of models they correspond
    to with the ability to easily recompute a model, what the bands are, what
    are the names of the properties, etc.

    """
    def __init__(self, conf, obs, params, iblock=0):
        self.conf = conf
        self.obs = obs
        self.params = params
        self.iblock = iblock
        self.block = params.blocks[iblock]

        self.propertiesnames = conf['analysis_params']['variables']
        self.allpropertiesnames, self.massproportional = self._get_info()

        self._fluxes_shape = (len(obs.bands), len(self.block))
        self._props_shape = (len(self.propertiesnames), len(self.block))

        # Arrays where we store the data related to the models. For memory
        # efficiency reasons, we use RawArrays that will be passed in argument
        # to the pool. Each worker will fill a part of the RawArrays. It is
        # important that there is no conflict and that two different workers do
        # not write on the same section.
        self._fluxes = self._shared_array(self._fluxes_shape)
        self._properties = self._shared_array(self._props_shape)

    @property
    def fluxes(self):
        """Returns a shared array containing the fluxes of the models.

        """
        return np.ctypeslib.as_array(self._fluxes).reshape(self._fluxes_shape)

    @property
    def properties(self):
        """Returns a shared array containing the properties of the models.

        """
        return np.ctypeslib.as_array(self._properties).reshape(self._props_shape)

    def _get_info(self):
        warehouse = SedWarehouse()
        sed = warehouse.get_sed(self.conf['sed_modules'],
                                self.params.from_index(0))
        info = list(sed.info.keys())
        info.sort()

        return (info, sed.mass_proportional_info)

    def save(self, filename):
        """Save the fluxes and properties of all the models into a table.

        Parameters
        ----------
        filename: str
            Root of the filename where to save the data.

        """
        table = Table(np.vstack((self.fluxes, self.properties)).T,
                      names=self.obs.bands + self.propertiesnames)

        table.add_column(Column(self.block, name='id'), index=0)

        table.write("out/{}.fits".format(filename))
        table.write("out/{}.txt".format(filename), format='ascii.fixed_width',
                    delimiter=None)

    @staticmethod
    def _shared_array(shape):
        return RawArray(ctypes.c_double, int(np.product(shape)))
