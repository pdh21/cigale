# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly
# Modified version by B. Lo Faro including the possibility of a delay-tau SFH + exponential burst
# modified on October 19th, 2016

"""
New Double star formation history module
===========================================================

This module implements a star formation history (SFH) composed of a delay-tau,
described as a delayed rise of the SFR up to a maximum followed by an exponential decrease,
 + a superimposed decreasing exponential burst.

"""

from collections import OrderedDict

import numpy as np

from . import SedModule


class sfhdelayedplusExpburst(SedModule):
    """Double Star Formation History: delay-tau + decreasing exponential

    """

    parameter_list = OrderedDict([
        ("tau_main", (
            "cigale_list()",
            "e-folding time of the main stellar population model in Myr.",
            3000.
        )),
        ("tau_burst", (
            "cigale_list()",
            "e-folding time of the late starburst population model in Myr.",
            10000.
        )),
        ("f_burst", (
            "cigale_list(minvalue=0., maxvalue=0.9999)",
            "Mass fraction of the late burst population.",
            [0.001, 0.010, 0.030, 0.100, 0.200, 0.300]
        )),
        ("age", (
            "cigale_list(dtype=int, minvalue=0.)",
            "Age of the main stellar population in the galaxy in Myr. The "
            "precision is 1 Myr.",
            [1000, 2500,  4500, 6000, 8000, 12000]
        )),
        ("burst_age", (
            "cigale_list(dtype=int, minvalue=1.)",
            "Age of the late burst in Myr. The precision is 1 Myr.",
            [10, 50, 80, 110]
        )),
        ("sfr_0", (
            "float(min=0)",
            "Value of SFR at t = 0 in M_sun/yr.",
            1.
        )),
        ("normalise", (
            "boolean()",
            "Normalise the SFH to produce one solar mass.",
            True
        )),
    ])

    def _init_code(self):
        self.tau_main = float(self.parameters["tau_main"])
        self.tau_burst = float(self.parameters["tau_burst"])
        self.f_burst = float(self.parameters["f_burst"])
        self.burst_age = int(self.parameters["burst_age"])
        age = int(self.parameters["age"])
        sfr_0 = float(self.parameters["sfr_0"])
        normalise = bool(self.parameters["normalise"])

        # Time grid and age. If needed, the age is rounded to the inferior Myr
        time_grid = np.arange(age)
        time_grid_burst = np.arange(self.burst_age)

        # SFR for each component
        self.sfr = time_grid * np.exp(-time_grid / self.tau_main) / \
                   self.tau_main**2
        sfr_burst = np.exp(-time_grid_burst / self.tau_burst)

        # Height of the late burst to have the desired produced mass fraction
        sfr_burst *= self.f_burst / (1.-self.f_burst) * np.sum(self.sfr) / np.sum(sfr_burst)

        # We add the age burst exponential for ages superior to age -
        # burst_age
        self.sfr[-(time_grid_burst[-1]+1):] += sfr_burst

        # Compute the galaxy mass and normalise the SFH to 1 solar mass
        # produced if asked to.
        self.sfr_integrated = np.sum(self.sfr) * 1e6
        if normalise:
            self.sfr /= self.sfr_integrated
            self.sfr_integrated = 1.
        else:
            self.sfr *= sfr_0
            self.sfr_integrated *= sfr_0

    def process(self, sed):
        """Add a double decreasing exponential Star Formation History.

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """

        sed.add_module(self.name, self.parameters)

        # Add the sfh and the output parameters to the SED.
        sed.sfh = self.sfr
        sed.add_info("sfh.integrated", self.sfr_integrated, True)
        sed.add_info("sfh.tau_main", self.tau_main)
        sed.add_info("sfh.tau_burst", self.tau_burst)
        sed.add_info("sfh.f_burst", self.f_burst)
        sed.add_info("sfh.burst_age", self.burst_age)

# SedModule to be returned by get_module
Module = sfhdelayedplusExpburst
