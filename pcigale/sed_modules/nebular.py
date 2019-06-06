# -*- coding: utf-8 -*-
# Copyright (C) 2014 University of Cambridge
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Médéric Boquien <mboquien@ast.cam.ac.uk>

from collections import OrderedDict
from copy import deepcopy

import numpy as np
import scipy.constants as cst

from pcigale.data import Database
from . import SedModule

default_lines = ['Ly-alpha',
                 'CII-133.5',
                 'SiIV-139.7',
                 'CIV-154.9',
                 'HeII-164.0',
                 'OIII-166.5',
                 'CIII-190.9',
                 'CII-232.6',
                 'MgII-279.8',
                 'OII-372.7',
                 'H-10',
                 'H-9',
                 'NeIII-386.9',
                 'HeI-388.9',
                 'H-epsilon',
                 'SII-407.0',
                 'H-delta',
                 'H-gamma',
                 'H-beta',
                 'OIII-495.9',
                 'OIII-500.7',
                 'OI-630.0',
                 'NII-654.8',
                 'H-alpha',
                 'NII-658.4',
                 'SII-671.6',
                 'SII-673.1'
                 ]

class NebularEmission(SedModule):
    """
    Module computing the nebular emission from the ultraviolet to the
    near-infrared. It includes both the nebular lines and the nubular
    continuum (optional). It takes into account the escape fraction and the
    absorption by dust.

    Given the number of Lyman continuum photons, we compute the Hβ line
    luminosity. We then compute the other lines using the
    metallicity-dependent templates that provide the ratio between individual
    lines and Hβ. The nebular continuum is scaled directly from the number of
    ionizing photons.

    """

    parameter_list = OrderedDict([
        ('logU', (
            'cigale_list(options=-4.0 & -3.9 & -3.8 & -3.7 & -3.6 & -3.5 & '
            '-3.4 & -3.3 & -3.2 & -3.1 & -3.0 & -2.9 & -2.8 & -2.7 & -2.6 & '
            '-2.5 & -2.4 & -2.3 & -2.2 & -2.1 & -2.0 & -1.9 & -1.8 & -1.7 & '
            '-1.6 & -1.5 & -1.4 & -1.3 & -1.2 & -1.1 & -1.0)',
            "Ionisation parameter",
            -2.
        )),
        ('f_esc', (
            'cigale_list(minvalue=0., maxvalue=1.)',
            "Fraction of Lyman continuum photons escaping the galaxy",
            0.
        )),
        ('f_dust', (
            'cigale_list(minvalue=0., maxvalue=1.)',
            "Fraction of Lyman continuum photons absorbed by dust",
            0.
        )),
        ('lines_width', (
            'cigale_list(minvalue=0.)',
            "Line width in km/s",
            300.
        )),
        ('emission', (
            'boolean()',
            "Include nebular emission.",
            True
        ))
    ])

    def _init_code(self):
        """Get the nebular emission lines out of the database and resample
           them to see the line profile. Compute scaling coefficients.
        """
        self.logU = float(self.parameters['logU'])
        self.fesc = float(self.parameters['f_esc'])
        self.fdust = float(self.parameters['f_dust'])
        self.lines_width = float(self.parameters['lines_width'])
        if type(self.parameters["emission"]) is str:
            self.emission = self.parameters["emission"].lower() == 'true'
        else:
            self.emission = bool(self.parameters["emission"])

        if self.fesc < 0. or self.fesc > 1:
            raise Exception("Escape fraction must be between 0 and 1")

        if self.fdust < 0 or self.fdust > 1:
            raise Exception("Fraction of lyman photons absorbed by dust must "
                            "be between 0 and 1")

        if self.fesc + self.fdust > 1:
            raise Exception("Escape fraction+f_dust>1")

        if self.emission:
            with Database() as db:
                metallicities = db.get_nebular_continuum_parameters()['metallicity']
                self.lines_template = {m: db.get_nebular_lines(m, self.logU)
                                    for m in metallicities}
                self.cont_template = {m: db.get_nebular_continuum(m, self.logU)
                                    for m in metallicities}

            self.linesdict = {m: dict(zip(self.lines_template[m].name,
                                          zip(self.lines_template[m].wave,
                                              self.lines_template[m].ratio)))
                              for m in metallicities}

            for lines in self.lines_template.values():
                new_wave = np.array([])
                for line_wave in lines.wave:
                    width = line_wave * self.lines_width * 1e3 / cst.c
                    new_wave = np.concatenate((new_wave,
                                            np.linspace(line_wave - 3. * width,
                                                        line_wave + 3. * width,
                                                        9)))
                new_wave.sort()
                new_flux = np.zeros_like(new_wave)
                for line_flux, line_wave in zip(lines.ratio, lines.wave):
                    width = line_wave * self.lines_width * 1e3 / cst.c
                    new_flux += (line_flux * np.exp(- 4. * np.log(2.) *
                                (new_wave - line_wave) ** 2. / (width * width)) /
                                (width * np.sqrt(np.pi / np.log(2.)) / 2.))
                lines.wave = new_wave
                lines.ratio = new_flux

            # To take into acount the escape fraction and the fraction of Lyman
            # continuum photons absorbed by dust we correct by a factor
            # k=(1-fesc-fdust)/(1+(α1/αβ)*(fesc+fdust))
            alpha_B = 2.58e-19  # Ferland 1980, m³ s¯¹
            alpha_1 = 1.54e-19  # αA-αB, Ferland 1980, m³ s¯¹
            k = (1. - self.fesc - self.fdust) / (1. + alpha_1 / alpha_B * (
                self.fesc + self.fdust))

            self.corr = k
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
            self.idx_Ly_break = np.searchsorted(sed.wavelength_grid, 91.2)
            self.absorbed_old = np.zeros(sed.wavelength_grid.size)
            self.absorbed_young = np.zeros(sed.wavelength_grid.size)

        self.absorbed_old[:self.idx_Ly_break] = -(
            sed.get_lumin_contribution('stellar.old')[:self.idx_Ly_break] *
            (1. - self.fesc))
        self.absorbed_young[:self.idx_Ly_break] = -(
            sed.get_lumin_contribution('stellar.young')[:self.idx_Ly_break] *
            (1. - self.fesc))

        sed.add_module(self.name, self.parameters)
        sed.add_info('nebular.f_esc', self.fesc)
        sed.add_info('nebular.f_dust', self.fdust)
        sed.add_info('dust.luminosity', (sed.info['stellar.lum_ly_young'] +
                     sed.info['stellar.lum_ly_old']) * self.fdust, True)

        sed.add_contribution('nebular.absorption_old', sed.wavelength_grid,
                             self.absorbed_old)
        sed.add_contribution('nebular.absorption_young', sed.wavelength_grid,
                             self.absorbed_young)

        if self.emission:
            NLy_old = sed.info['stellar.n_ly_old']
            NLy_young = sed.info['stellar.n_ly_young']
            NLy_tot = NLy_old + NLy_young
            metallicity = sed.info['stellar.metallicity']
            lines = self.lines_template[metallicity]
            linesdict = self.linesdict[metallicity]
            cont = self.cont_template[metallicity]

            sed.add_info('nebular.lines_width', self.lines_width)
            sed.add_info('nebular.logU', self.logU)

            for line in default_lines:
                wave, ratio = linesdict[line]
                sed.lines[line] = (wave,
                                   ratio * NLy_old * self.corr,
                                   ratio * NLy_young * self.corr)

            sed.add_contribution('nebular.lines_old', lines.wave,
                                 lines.ratio * NLy_old * self.corr)
            sed.add_contribution('nebular.lines_young', lines.wave,
                                 lines.ratio * NLy_young * self.corr)

            sed.add_contribution('nebular.continuum_old', cont.wave,
                                 cont.lumin * NLy_old * self.corr)
            sed.add_contribution('nebular.continuum_young', cont.wave,
                                 cont.lumin * NLy_young * self.corr)


# SedModule to be returned by get_module
Module = NebularEmission
