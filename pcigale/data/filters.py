# -*- coding: utf-8 -*-
# Copyright (C) 2012 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly

import numpy as np


class Filter(object):
    """A photometric filter with associated transmission data.
    """

    def __init__(self, name, description=None, trans_table=None,
                 effective_wavelength=None):
        """Create a new filter. If the transmission type, the description
        the transmission table or the effective wavelength are not specified,
        their value is set to None.

        Parameters
        ----------
        name: string
            Name of the filter
        description: string
            Description of the filter
        trans_table: array
            trans_table[0] is the wavelength in nm,
            trans_table[1] is the transmission)
        effective_wavelength: float
            Effective wavelength of the filter
        """

        self.name = name
        self.description = description
        self.trans_table = trans_table
        self.effective_wavelength = effective_wavelength

    def __str__(self):
        """Pretty print the filter information
        """
        result = ""
        result += ("Filter name: %s\n" % self.name)
        result += ("Description: %s\n" % self.description)
        result += ("Effective wavelength: %s nm\n" %
                   self.effective_wavelength)
        return result

    def normalise(self):
        """
        Normalise the transmission table to 1 and compute the effective
        wavelength of the filter.
        """
        self.trans_table[1] = self.trans_table[1] / (
            np.trapz(self.trans_table[1], self.trans_table[0]))

        self.effective_wavelength = np.trapz(self.trans_table[1] *
                                             self.trans_table[0],
                                             self.trans_table[0])
