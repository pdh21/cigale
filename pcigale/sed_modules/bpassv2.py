# -*- coding: utf-8 -*-
# Copyright (C) 2017 Universidad de Antofagasta
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Médéric Boquien

"""
BPASS v2 stellar emission module
==================================================

This module implements the BPASS v2 Single Stellar Populations.

"""

from collections import OrderedDict

import numpy as np

from . import SedModule
from ..data import Database


class BPASSv2(SedModule):
    """BPASS v2 stellar emission module

    This SED creation module convolves the SED star formation history with a
    BPASS v2 single stellar population to add a stellar component to the SED.

    """

    parameter_list = OrderedDict([
        ("imf", (
            "cigale_list(dtype=int, options=0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8)",
            "Initial mass function: 0 (-1.30 between 0.1 to 0.5Msun and -2.00 "
            "from 0.5 to 300Msun), 1 (-1.30 between 0.1 to 0.5Msun and -2.00 "
            "from 0.5 to 100Msun), 2 (-2.35 from 0.1 to 100Msun), 3 (-1.30 "
            "between 0.1 to 0.5Msun and -2.35 from 0.5 to 300Msun), 4 (-1.30 "
            "between 0.1 to 0.5Msun and -2.35 from 0.5 to 100Msun), 5 (-1.30 "
            "between 0.1 to 0.5Msun and -2.70 from 0.5 to 300Msun), 6 (-1.30 "
            "between 0.1 to 0.5Msun and -2.70 from 0.5 to 100Msun), 7 ("
            "Chabrier up to 100Msun), 8 (Chabrier up to 300Msun).",
            2
        )),
        ("metallicity", (
            "cigale_list(options=0.001 & 0.002 & 0.003 & 0.004 & 0.006 & "
            "0.008 & 0.010 & 0.014 & 0.020 & 0.030 & 0.040)",
            "Metalicity. Possible values are: 0.001, 0.002, 0.003, 0.004, "
            "0.006, 0.008, 0.010, 0.014, 0.020, 0.030, 0.040.",
            0.02
        )),
        ("binary", (
            "cigale_list(options=0 & 1)",
            "Single (0) or binary (1) stellar populations.",
            0
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
        self.separation_age = int(self.parameters["separation_age"])
        self.binary = bool(self.parameters["binary"])

        with Database() as db:
            self.ssp = db.get_bpassv2(self.imf, self.metallicity, self.binary)

        self.wave = self.ssp.wavelength_grid
        self.w_lymanc = np.where(self.wave <= 91.1)

    def process(self, sed):
        """Add the convolution of a Bruzual and Charlot SSP to the SED

        Parameters
        ----------
        sed: pcigale.sed.SED
            SED object.

        """
        out = self.ssp.convolve(sed.sfh, self.separation_age)
        spec_young, spec_old, info_young, info_old, info_all = out

        # We compute the Lyman continuum luminosity as it is important to
        # compute the energy absorbed by the dust before ionising gas.
        wave_lymanc = self.wave[self.w_lymanc]
        lum_lyc_young = np.trapz(spec_young[self.w_lymanc], wave_lymanc)
        lum_lyc_old = np.trapz(spec_old[self.w_lymanc], wave_lymanc)

        # We do similarly for the total stellar luminosity
        lum_young, lum_old = np.trapz([spec_young, spec_old], self.wave)

        sed.add_module(self.name, self.parameters)

        sed.add_info("stellar.imf", self.imf)
        sed.add_info("stellar.metallicity", self.metallicity)
        sed.add_info("stellar.binary", float(self.binary))

        sed.add_info("stellar.m_star_young", info_young["m_star"], True)
        sed.add_info("stellar.n_ly_young", info_young["n_ly"], True)
        sed.add_info("stellar.lum_ly_young", lum_lyc_young, True)
        sed.add_info("stellar.lum_young", lum_young, True)

        sed.add_info("stellar.m_star_old", info_old["m_star"], True)
        sed.add_info("stellar.n_ly_old", info_old["n_ly"], True)
        sed.add_info("stellar.lum_ly_old", lum_lyc_old, True)
        sed.add_info("stellar.lum_old", lum_old, True)

        sed.add_info("stellar.m_star", info_all["m_star"], True)
        sed.add_info("stellar.n_ly", info_all["n_ly"], True)
        sed.add_info("stellar.lum", lum_young + lum_old, True)

        sed.add_contribution("stellar.old", self.wave, spec_old)
        sed.add_contribution("stellar.young", self.wave, spec_young)


# SedModule to be returned by get_module
Module = BPASSv2
