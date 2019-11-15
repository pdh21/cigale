# -*- coding: utf-8 -*-
# Copyright (C) 2012, 2013 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly

"""
Maraston (2005) stellar emission module
=======================================

This module implements the Maraston (2005) Single Stellar Populations.

"""

from collections import OrderedDict

import numpy as np

from . import SedModule
from ..data import Database


class M2005(SedModule):
    """Maraston (2005) stellar emission module

    This SED creation module convolves the SED star formation history with
    a Maraston (2005) single stellar population to add a stellar component to
    the SED.

    """

    parameter_list = OrderedDict([
        ('imf', (
            'cigale_list(dtype=int, options=0. & 1.)',
            "Initial mass function: 0 (Salpeter) or 1 (Kroupa)",
            0
        )),
        ('metallicity', (
            'cigale_list(options=0.001 & 0.01 & 0.02 & 0.04)',
            "Metallicity. Possible values are: 0.001, 0.01, 0.02, 0.04.",
            0.02
        )),
        ('separation_age', (
            'cigale_list(dtype=int, minvalue=0.)',
            "Age [Myr] of the separation between the young and the old star "
            "populations. The default value in 10^7 years (10 Myr). Set to "
            "0 not to differentiate ages (only an old population).",
            10
        ))
    ])

    def _init_code(self):
        """Read the SSP from the database."""
        self.imf = int(self.parameters["imf"])
        self.metallicity = float(self.parameters["metallicity"])
        self.separation_age = int(self.parameters["separation_age"])

        if self.imf == 0:
            with Database() as database:
                self.ssp = database.get_m2005('salp', self.metallicity)
        elif self.imf == 1:
            with Database() as database:
                self.ssp = database.get_m2005('krou', self.metallicity)
        else:
            raise Exception(f"IMF #{self.imf} unknown")

    def process(self, sed):
        """Add the convolution of a Maraston 2005 SSP to the SED

        Parameters
        ----------
        sed: pcigale.sed.SED
            SED object.

        """
        out = self.ssp.convolve(sed.sfh, self.separation_age)
        spec_young, spec_old, info_young, info_old, info_all = out
        lum_young, lum_old = np.trapz([spec_young, spec_old],
                                      self.ssp.wavelength_grid)

        sed.add_module(self.name, self.parameters)

        sed.add_info('stellar.imf', self.imf)
        sed.add_info('stellar.metallicity', self.metallicity)
        sed.add_info('stellar.old_young_separation_age', self.separation_age)

        sed.add_info('stellar.mass_total_young', info_young[0], True)
        sed.add_info('stellar.mass_alive_young', info_young[1], True)
        sed.add_info('stellar.mass_white_dwarf_young', info_young[2], True)
        sed.add_info('stellar.mass_neutron_young', info_young[3], True)
        sed.add_info('stellar.mass_black_hole_young', info_young[4], True)
        sed.add_info('stellar.lum_young', lum_young, True)

        sed.add_info('stellar.mass_total_old', info_old[0], True)
        sed.add_info('stellar.mass_alive_old', info_old[1], True)
        sed.add_info('stellar.mass_white_dwarf_old', info_old[2], True)
        sed.add_info('stellar.mass_neutron_old', info_old[3], True)
        sed.add_info('stellar.mass_black_hole_old', info_old[4], True)
        sed.add_info('stellar.lum_old', lum_old, True)

        sed.add_info('stellar.mass_total', info_all[0], True)
        sed.add_info('stellar.mass_alive', info_all[1], True)
        sed.add_info('stellar.mass_white_dwarf', info_all[2], True)
        sed.add_info('stellar.mass_neutron', info_all[3], True)
        sed.add_info('stellar.mass_black_hole', info_all[4], True)
        sed.add_info('stellar.age_mass', info_all[5])
        sed.add_info('stellar.lum', lum_young + lum_old, True)

        sed.add_contribution("stellar.old", self.ssp.wavelength_grid, spec_old)
        sed.add_contribution("stellar.young", self.ssp.wavelength_grid,
                             spec_young)


# SedModule to be returned by get_module
Module = M2005
