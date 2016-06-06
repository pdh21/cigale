# -*- coding: utf-8 -*-
# Copyright (C) 2014 Yannick Roehlly, Médéric Boquien, Denis Burgarella
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly, Médéric Boquien, Denis Burgarella

"""
Module that sets the redshift
=============================

This module is specifically designed for simulations where the amount of dust attenuation is redshift-dependent. So, we need to read it before the dust attenuation module.

"""

from collections import OrderedDict

import numpy as np
from scipy.constants import parsec
from astropy.cosmology import WMAP7 as cosmology
import astropy.units as u
import scipy.constants as cst

from . import SedModule

class set_redshift(SedModule):
    """Redshift a SED

    This module reads the redshifts that will be used to create the models, but,
    also to estimate the amount of dust attenuation via the A_FUV (M_star) relation.

    """

    parameter_list = OrderedDict([
        ("redshift", (
            "cigale_list(minvalue=0.)",
            "\"eval np.linspace(z_min, z_max, N_z)\" (warning: z_min > 0.5*step_z)",
            "eval np.linspace(0., 20., 100)"
        ))
    ])

    def _init_code(self):
        """Compute the age of the Universe at a given redshift
        """

        self.redshift = float(self.parameters["redshift"])

        # Raise an error when applying a negative redshift. This module is
        # not for blue-shifting.
        if self.redshift < 0.:
            raise Exception("The redshift provided is negative <{}>."
                            .format(self.redshift))

        self.universe_age = cosmology.age(self.redshift).value * 1000.
        if self.redshift == 0.:
            self.luminosity_distance = 10. * parsec
        else:
            self.luminosity_distance = (
                cosmology.luminosity_distance(self.redshift).value * 1e6 *
                parsec)

    def process(self, sed):
        """Redshift the SED

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """
        redshift = self.redshift

        # If the redshift is already stored, raise an error.
        if ('universe.redshift' in sed.info and
            sed.info['universe.redshift'] > 0.):
            raise Exception("The redshift is already stored <z={}>."
                            .format(sed.info['universe.redshift']))

        sed.add_info("universe.redshift", redshift)
        sed.add_info("universe.luminosity_distance", self.luminosity_distance)
        sed.add_info("universe.age", self.universe_age)

        sed.add_module(self.name, self.parameters)

# SedModule to be returned by get_module
Module = set_redshift
