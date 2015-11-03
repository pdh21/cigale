# -*- coding: utf-8 -*-
# Copyright (C) 2014 University of Cambridge
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Médéric Boquien <mboquien@ast.cam.ac.uk>

from collections import OrderedDict

import numpy as np
import scipy.constants as cst

from pcigale.data import Database
from . import CreationModule


class NebularEmission(CreationModule):
    """
    Module computing the nebular emission from the ultraviolet to the
    near-infrared. It includes both the nebular lines and the nubular
    continuum. It takes into account the escape fraction and the absorption by
    dust.

    Given the number of Lyman continuum photons, we compute the Hβ line
    luminosity. We then compute the other lines using the
    metallicity-dependent templates that provide the ratio between individual
    lines and Hβ. The nebular continuum is scaled directly from the number of
    ionizing photons.

    """

    parameter_list = OrderedDict([
        ('f_esc', (
            'float',
            "Fraction of Lyman continuum photons escaping the galaxy",
            0.
        )),
        ('f_dust', (
            'float',
            "Fraction of Lyman continuum photons absorbed by dust",
            0.
        ))
    ])

    def _init_code(self):
        """Get the nebular emission lines out of the database and resample
           them to see the line profile. Compute scaling coefficients.
        """

        fesc = self.parameters['f_esc']
        fdust = self.parameters['f_dust']

        if fesc < 0. or fesc > 1:
            raise Exception("Escape fraction must be between 0 and 1")

        if fdust < 0 or fdust > 1:
            raise Exception("Fraction of lyman photons absorbed by dust must "
                            "be between 0 and 1")

        if fesc + fdust > 1:
            raise Exception("Escape fraction+f_dust>1")

        self.idx_Ly_break = None
        self.absorbed_old = None
        self.absorbed_young = None

    def process(self, sed):
        """Add the nebular emission lines

        Parameters
        ----------
        sed: pcigale.sed.SED object
        parameters: dictionary containing the parameters

        """
        if self.idx_Ly_break is None:
            self.idx_Ly_break = np.searchsorted(sed.wavelength_grid, 91.)
            self.absorbed_old = np.zeros(sed.wavelength_grid.size)
            self.absorbed_young = np.zeros(sed.wavelength_grid.size)
        self.absorbed_old[:self.idx_Ly_break] -= sed.get_lumin_contribution('stellar.old')[:self.idx_Ly_break] * (1. - self.parameters['f_esc'])
        self.absorbed_young[:self.idx_Ly_break] -= sed.get_lumin_contribution('stellar.young')[:self.idx_Ly_break] * (1. - self.parameters['f_esc'])

        sed.add_module(self.name, self.parameters)
        sed.add_info('nebular.f_esc', self.parameters['f_esc'])
        sed.add_info('nebular.f_dust', self.parameters['f_dust'])
        sed.add_info('dust.luminosity', (sed.info['stellar.lum_ly_young'] +
                     sed.info['stellar.lum_ly_old']) *
                     self.parameters['f_dust'], True)

        sed.add_contribution('nebular.absorption_old', sed.wavelength_grid,
                             self.absorbed_old)
        sed.add_contribution('nebular.absorption_young', sed.wavelength_grid,
                             self.absorbed_young)


# CreationModule to be returned by get_module
Module = NebularEmission
