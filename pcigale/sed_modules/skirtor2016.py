# -*- coding: utf-8 -*-
# Copyright (C) 2013, 2014 Department of Physics, University of Crete
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Laure Ciesla

"""
SKIRTOR 2016 (Stalevski et al., 2016) AGN dust torus emission module
==================================================

This module implements the SKIRTOR 2016 models.

"""
from collections import OrderedDict

import numpy as np

from pcigale.data import Database
from . import SedModule


class SKIRTOR2016(SedModule):
    """SKIRTOR 2016 (Stalevski et al., 2016) AGN dust torus emission


    The relative normalization of these components is handled through a
    parameter which is the fraction of the total IR luminosity due to the AGN
    so that: L_AGN = fracAGN * L_IRTOT, where L_AGN is the AGN luminosity,
    fracAGN is the contribution of the AGN to the total IR luminosity
    (L_IRTOT), i.e. L_Starburst+L_AGN.

    """

    parameter_list = OrderedDict([
        ('t', (
            "cigale_list(options=3 & 5 & 7 & 9 & 11)",
            "Average edge-on optical depth at 9.7 micron; the actual one along"
            "the line of sight may vary depending on the clumps distribution. "
            "Possible values are: 3, 5, 7, 8, and 11.",
            3
        )),
        ('pl', (
            "cigale_list(options=0. & .5 & 1. & 1.5)",
            "Power-law exponent that sets radial gradient of dust density."
            "Possible values are: 0., 0.5, 1., and 1.5.",
            1.0
        )),
        ('q', (
            "cigale_list(options=0. & .5 & 1. & 1.5)",
            "Index that sets dust density gradient with polar angle."
            "Possible values are:  0., 0.5, 1., and 1.5.",
            1.0
        )),
        ('oa', (
            'cigale_list(options=10 & 20 & 30 & 40 & 50 & 60 & 70 & 80)',
            "Angle measured between the equatorial plan and edge of the torus. "
            "Half-opening angle of the dust-free cone is 90-oa"
            "Possible values are: 10, 20, 30, 40, 50, 60, 70, and 80",
            40
        )),
        ('R', (
            'cigale_list(options=10 & 20 & 30)',
            "Ratio of outer to inner radius, R_out/R_in."
            "Possible values are: 10, 20, and 30",
            20
        )),
        ('Mcl', (
            'cigale_list(options=0.97)',
            "fraction of total dust mass inside clumps. 0.97 means 97% of "
            "total mass is inside the clumps and 3% in the interclump dust. "
            "Possible values are: 0.97.",
            0.97
        )),
        ('i', (
            'cigale_list(options=0 & 10 & 20 & 30 & 40 & 50 & 60 & 70 & 80 & 90)',
            "inclination, i.e. viewing angle, i.e. position of the instrument "
            "w.r.t. the AGN axis. i=0: face-on, type 1 view; i=90: edge-on, "
            "type 2 view."
            "Possible values are: 0, 10, 20, 30, 40, 50, 60, 70, 80, and 90.",
            40
        )),
        ('fracAGN', (
            'cigale_list(minvalue=0., maxvalue=1.)',
            "AGN fraction.",
            0.1
        ))
    ])

    def _init_code(self):
        """Get the template set out of the database"""
        self.t = int(self.parameters["t"])
        self.pl = float(self.parameters["pl"])
        self.q = float(self.parameters["q"])
        self.oa = int(self.parameters["oa"])
        self.R = int(self.parameters["R"])
        self.Mcl = float(self.parameters["Mcl"])
        self.i = int(self.parameters["i"])
        self.fracAGN = float(self.parameters["fracAGN"])

        with Database() as base:
            self.SKIRTOR2016 = base.get_skirtor2016(self.t, self.pl, self.q,
                                                    self.oa, self.R, self.Mcl,
                                                    self.i)

    def process(self, sed):
        """Add the IR re-emission contributions

        Parameters
        ----------
        sed: pcigale.sed.SED object
        parameters: dictionary containing the parameters

        """

        if 'dust.luminosity' not in sed.info:
            sed.add_info('dust.luminosity', 1., True)
        luminosity = sed.info['dust.luminosity']

        sed.add_module(self.name, self.parameters)
        sed.add_info('agn.t', self.t)
        sed.add_info('agn.pl', self.pl)
        sed.add_info('agn.q', self.q)
        sed.add_info('agn.oa', self.oa)
        sed.add_info('agn.R', self.R)
        sed.add_info('agn.Mcl', self.Mcl)
        sed.add_info('agn.i', self.i)
        sed.add_info('agn.fracAGN', self.fracAGN)

        # Compute the AGN luminosity
        if self.fracAGN < 1.:
            agn_power = luminosity * (1./(1.-self.fracAGN) - 1.)
            lumin_dust = agn_power
            lumin_disk = agn_power * np.trapz(self.SKIRTOR2016.disk,
                                              x=self.SKIRTOR2016.wave)
            lumin = lumin_dust + lumin_disk
        else:
            raise Exception("AGN fraction is exactly 1. Behaviour "
                            "undefined.")

        sed.add_info('agn.dust_luminosity', lumin_dust, True)
        sed.add_info('agn.disk_luminosity', lumin_disk, True)
        sed.add_info('agn.luminosity', lumin, True)

        sed.add_contribution('agn.SKIRTOR2016_dust', self.SKIRTOR2016.wave,
                             agn_power * self.SKIRTOR2016.dust)
        sed.add_contribution('agn.SKIRTOR2016_disk', self.SKIRTOR2016.wave,
                             agn_power * self.SKIRTOR2016.disk)


# SedModule to be returned by get_module
Module = SKIRTOR2016
