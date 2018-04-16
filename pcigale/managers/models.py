# -*- coding: utf-8 -*-
# Copyright (C) 2017 Universidad de Antofagasta
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Médéric Boquien

"""This class manages the handling of models in the code. It contains the
fluxes and the physical properties and all the necessary information to
compute them, such as the configuration, the observations, and the parameters
of the models.
"""

from astropy.table import Table, Column

from .utils import SharedArray, get_info


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
        self.block = params.blocks[iblock]
        self.iblock = iblock
        self.allpropnames, self.allextpropnames = get_info(self)
        self.allintpropnames = set(self.allpropnames) - self.allextpropnames

        props_nolog = set([prop[:-4] if prop.endswith('log') else prop
                           for prop in conf['analysis_params']['variables']])
        self.intpropnames = (self.allintpropnames & set(obs.intprops) |
                             self.allintpropnames & props_nolog)
        self.extpropnames = (self.allextpropnames & set(obs.extprops) |
                             self.allextpropnames & props_nolog)
        size = len(params.blocks[iblock])

        self.flux = {band: SharedArray(size) for band in obs.bands}
        self.intprop = {prop: SharedArray(size) for prop in self.intpropnames}
        self.extprop = {prop: SharedArray(size) for prop in self.extpropnames}

    @property
    def flux(self):
        """Returns a shared array containing the fluxes of the models.

        """
        return self._flux

    @flux.setter
    def flux(self, flux):
        self._flux = flux

    @property
    def intprop(self):
        """Returns a shared array containing the intensive properties to fit.
        """
        return self._intprop

    @intprop.setter
    def intprop(self, intprop):
        self._intprop = intprop

    @property
    def extprop(self):
        """Returns a shared array containing the extensive properties to fit.
        """
        return self._extprop

    @extprop.setter
    def extprop(self, extprop):
        self._extprop = extprop

    def save(self, filename):
        """Save the fluxes and properties of all the models into a table.

        Parameters
        ----------
        filename: str
            Root of the filename where to save the data.

        """
        table = Table()
        table.add_column(Column(self.block, name='id'))
        for band in sorted(self.flux.keys()):
            table.add_column(Column(self.flux[band], name=band,
                                    unit='mJy'))
        for prop in sorted(self.extprop.keys()):
            table.add_column(Column(self.extprop[prop], name=prop))
        for prop in sorted(self.intprop.keys()):
            table.add_column(Column(self.intprop[prop], name=prop))

        table.write("out/{}.fits".format(filename))
        table.write("out/{}.txt".format(filename), format='ascii.fixed_width',
                    delimiter=None)
