# -*- coding: utf-8 -*-
# Copyright (C) 2017 University of Sussex
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly

"""
Simple module to add Lymann continuum absorption
================================================

This module is a simple module developed for the Herschel Extragalactic Legacy
Programme (HELP) to deal with the Lymann continuum without using the full
nebular module from CIGALE.

"""

from collections import OrderedDict

import numpy as np

from . import SedModule


class LycAbsorption(SedModule):
    """Module to remove the Lymann continuum.

    This module reduces the emission in the Lymann continuum.  It uses two
    parameters:

    - f_esc is the fraction of Lymann continuum photons escaping the galaxy;
    - f_dust is the fraction of these photons that heat the dust.

    The remaining photons are supposed to be capture by gas and to be
    re-emitted as emission lines but are not dealt with by this module.

    """

    parameter_list = OrderedDict([
        ('f_esc', (
            'cigale_list(minvalue=0., maxvalue=1.)',
            "Fraction of Lyman continuum photons escaping the galaxy",
            0.
        )),
        ('f_dust', (
            'cigale_list(minvalue=0., maxvalue=1.)',
            "Fraction of Lyman continuum photons absorbed by dust",
            0.
        ))
    ])

    def _init_code(self):
        self.f_esc = float(self.parameters['f_esc'])
        self.f_dust = float(self.parameters['f_dust'])

        if self.f_esc + self.f_dust > 1:
            raise ValueError("Sum of f_esc and f_dust must be lower than 1.")

    def process(self, sed):
        """Reduce the Lymann continuum emission.

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """
        wavelength = sed.wavelength_grid
        luminosity = sed.luminosity
        lyc_mask = wavelength <= 91.2

        sed.add_module(self.name, self.parameters)
        sed.add_info("lyc_absorption_f_esc", self.f_esc)
        sed.add_info("lyc_absorption_f_dust", self.f_dust)

        if self.f_dust > 0:
            lyc_dust_lumin = self.f_dust * np.trapz(luminosity[lyc_mask],
                                                    wavelength[lyc_mask])
            # We add the luminosity absorbed by the dust to the global dust
            # luminosity of the SED, if the SED has no dust luminosity, we
            # first add it to 0.
            if 'dust_luminosity' not in sed.info:
                sed.add_info("dust_luminosity", 0., mass_proportional=True)
            sed.add_info("dust_luminosity",
                         sed.info['dust_luminosity'] + lyc_dust_lumin,
                         mass_proportional=True,
                         force=True)

        lyc_absorption = np.zeros_like(wavelength)
        lyc_absorption[lyc_mask] = -(sed.luminosity[lyc_mask] *
                                     (1. - self.f_esc))
        sed.add_contribution("lyc_absorption", wavelength, lyc_absorption)


# CreationModule to be returned by get_module
Module = LycAbsorption
