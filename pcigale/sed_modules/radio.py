# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Copyright (C) 2013 Institute of Astronomy, University of Cambridge
# Copyright (C) 2014 University of Crete
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Laure Ciesla & Médéric Boquien

"""
Radio module
=============================

This module implements the synchrotron emission of galaxies assuming the
FIR/radio correlation and the power law of the synchrotron spectrum.

"""

from collections import OrderedDict

import numpy as np
import scipy.constants as cst

from . import SedModule


class Radio(SedModule):
    """Radio emission

    This module computes the synchrotron (non-thermal) emission of galaxies.

    """

    parameter_list = OrderedDict([
        ("qir", (
            "cigale_list(minvalue=0.)",
            "The value of the FIR/radio correlation coefficient.",
            2.58
        )),
        ("alpha", (
            "cigale_list()",
            "The slope of the power-law synchrotron emission.",
            0.8
        ))
    ])

    def _init_code(self):
        """Build the model for a given set of parameters."""

        self.qir = float(self.parameters["qir"])
        self.alpha = float(self.parameters["alpha"])

        # We define various constants necessary to compute the model. For
        # consistency, we define the speed of light in nm s¯¹ rather than in
        # m s¯¹.
        c = cst.c * 1e9
        # We define the wavelength range for the non thermal emission
        self.wave = np.logspace(5., 9., 1000.)
        # We compute the synchrotron emission normalised at 21cm
        self.lumin_nonthermal = ((1./self.wave)**(-self.alpha + 2.) /
                                 (1./2.1e8)**(-self.alpha + 2.))
        # Normalisation factor from the FIR/radio correlation to apply to the
        # IR luminosity
        S21cm = (1. / (10.**self.qir*3.75e12)) * (c/(2.1e8)**2)
        self.lumin_nonthermal *= S21cm

    def process(self, sed):
        """Add the radio contribution.

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """
        if 'dust.luminosity' not in sed.info:
            sed.add_info('dust.luminosity', 1., True)
        luminosity = sed.info['dust.luminosity']

        sed.add_module(self.name, self.parameters)
        sed.add_info("radio.qir", self.qir)
        sed.add_info("radio.alpha", self.alpha)
        sed.add_contribution('radio_nonthermal', self.wave,
                             self.lumin_nonthermal * luminosity)


# SedModule to be returned by get_module
Module = Radio