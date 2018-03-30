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
        self.iblock = iblock
        self.block = params.blocks[iblock]

        self.propertiesnames = conf['analysis_params']['variables']
        self.allpropertiesnames, self.massproportional = get_info(self)

        self._fluxes = SharedArray((len(self.obs.bands), len(self.block)))
        self._properties = SharedArray((len(self.propertiesnames),
                                        len(self.block)))

        if conf['analysis_method'] == 'pdf_analysis':
            self._intprops = SharedArray((len(self.obs.intprops),
                                          len(self.block)))
            self._extprops = SharedArray((len(self.obs.extprops),
                                          len(self.block)))

    @property
    def fluxes(self):
        """Returns a shared array containing the fluxes of the models.

        """
        return self._fluxes.array

    @property
    def properties(self):
        """Returns a shared array containing the properties of the models.

        """
        return self._properties.array

    @property
    def intprops(self):
        """Returns a shared array containing the intensive properties to fit.
        """
        return self._intprops.array

    @property
    def extprops(self):
        """Returns a shared array containing the extensive properties to fit.
        """
        return self._extprops.array

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
