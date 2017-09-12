# -*- coding: utf-8 -*-
# Copyright (C) 2014 Institute of Astronomy
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Médéric Boquien

"""
Starburst 99 stellar emission module
==================================================

This module implements the Starburst 99 Single Stellar
Populations.

"""

import numpy as np
from collections import OrderedDict
from . import SedModule
from ..data import Database


class SB99(SedModule):
    """Starburst99 stellar emission module

    This SED creation module convolves the SED star formation history with a
    Starburst 99 single stellar population to add a stellar component to the
    SED.
    """

    parameter_list = OrderedDict([
        ("imf", (
            "cigale_list(dtype=int, options=0 & 1)",
            "Initial mass function: 0 (Salpeter) or 1 (Kroupa).",
            1
        )),
        ("metallicity", (
            "cigale_list(options=0.002 & 0.014)",
            "Metalicity. Possible values are: 0.002, 0.014.",
            0.014
        )),
        ("rotation", (
            "cigale_list(options=0.0 & 0.4)",
            "Stellar rotation. Possible values are: 0., 0.4.",
            0.
        )),
        ("separation_age", (
            "cigale_list(dtype=int, minvalue=0)",
            "Age [Myr] of the separation between the young and the old star "
            "populations. The default value in 10^7 years (10 Myr). Set "
            "to 0 not to differentiate ages (only an old population).",
            10
        ))
    ])

    def _init_code(self):
        """Read the SSP from the database."""
        self.imf = int(self.parameters["imf"])
        self.metallicity = float(self.parameters["metallicity"])
        self.rotation = float(self.parameters["rotation"])
        self.separation_age = int(self.parameters["separation_age"])
        with Database() as database:
            if self.imf == 0:
                self.ssp = database.get_sb99('salp', self.metallicity,
                                             self.rotation)
            elif self.imf == 1:
                self.ssp = database.get_sb99('krou', self.metallicity,
                                             self.rotation)
            else:
                raise Exception("IMF #{} unknown".format(self.imf))

    def process(self, sed):
        """Add the convolution of a Starburst 99 SSP to the SED

        Parameters
        ----------
        sed: pcigale.sed.SED
            SED object.

        """
        out = self.ssp.convolve(sed.sfh, self.separation_age)
        spec_young, spec_old, info_young, info_old, info_all = out

        sed.add_module(self.name, self.parameters)

        sed.add_info("stellar.imf", self.imf)
        sed.add_info("stellar.metallicity", self.metallicity)
        sed.add_info("stellar.rotation", self.rotation)
        sed.add_info("stellar.old_young_separation_age", self.separation_age)

        sed.add_info("stellar.m_star_young", info_young["m_star"], True)
        sed.add_info("stellar.n_ly_young", info_young["n_ly"], True)

        sed.add_info("stellar.m_star_old", info_old["m_star"], True)
        sed.add_info("stellar.n_ly_old", info_old["n_ly"], True)

        sed.add_contribution("stellar.old", self.ssp.wavelength_grid, spec_old)
        sed.add_contribution("stellar.young", self.ssp.wavelength_grid,
                             spec_young)

# CreationModule to be returned by get_module
Module = SB99
