# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Copyright (C) 2013 Institute of Astronomy, University of Cambridge
# Copyright (C) 2014 University of Crete
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Guang Yang

"""
X-ray module
=============================

This module implements the X-ray emission from the galaxy and AGN corona.

"""

from collections import OrderedDict

import numpy as np
import scipy.constants as cst

from . import SedModule

class Xray(SedModule):
    """X-ray emission

    This module computes the X-ray emission from the galaxy and AGN corona.

    """

    parameter_list = OrderedDict([
        ("gam", (
            "cigale_list()",
            "The photon index (Gamma) of AGN intrinsic X-ray spectrum.",
            1.8
        )),
        ("max_dev_alpha_ox", (
            "cigale_list()",
            "Maximum deviation of alpha_ox from the empirical alpha_ox-Lnu_2500A relation (Just et al. 2007), "
            "i.e. |alpha_ox-alpha_ox(Lnu_2500A)| <= max_dev_alpha_ox. "
            "alpha_ox is the power-law slope connecting L_nu at rest-frame 2500 A and 2 keV, "
            "defined as alpha_ox = 0.3838*log(Lnu_2keV/Lnu_2500A), "
            "which is often modeled as a function of Lnu_2500A. "
            "The alpha_ox-Lnu_2500A relation has a 1-sigma scatter of ~ 0.1.",
            0.2
        )),
        ("gam_lmxb", (
            "cigale_list()",
            "The photon index of AGN low-mass X-ray binaries.",
            1.56
        )),
        ("gam_hmxb", (
            "cigale_list()",
            "The photon index of AGN high-mass X-ray binaries.",
            2.0
        ))
    ])

    def _init_code(self):
        """Build the model for a given set of parameters."""

        self.gam = float(self.parameters["gam"])
        self.gam_lmxb = float(self.parameters["gam_lmxb"])
        self.gam_hmxb = float(self.parameters["gam_hmxb"])
        self.max_dev_alpha_ox = float(self.parameters["max_dev_alpha_ox"])

        # We define various constants necessary to compute the model. For
        # consistency, we define speed of light in units of nm s¯¹
        c = cst.c * 1e9
        self.c = c
        # Define wavelenght corresponding to some energy in units of nm.
        lam_1keV = c*cst.h / (1e3*cst.eV)
        lam_0p5keV = lam_1keV*2
        lam_100keV = lam_1keV/100
        lam_300keV = lam_1keV/300
        self.lam_2keV = lam_1keV/2
        self.lam_10keV = lam_1keV/10
        # Define frequency corresponding to 2 keV in units of Hz.
        self.nu_2keV = c / self.lam_2keV

        # We define the wavelength grid for the X-ray emission
        # corresponding to 0.25-1200 keV
        self.wave = np.logspace(-3, 0.7, 1000)

        # X-ray emission from galaxies: 1.hot-gas & 2.X-ray binaries
        # 1.Hot-gas, assuming power-law index gamma=1, E_cut=1 keV
        # normalized such that L(0.5-2 keV) = 1
        self.lumin_hotgas = self.wave**-2 * np.exp(-lam_1keV/self.wave)
        lam_idxs = (self.wave<=lam_0p5keV) & (self.wave>=self.lam_2keV)
        self.lumin_hotgas /= np.trapz(self.lumin_hotgas[lam_idxs], x=self.wave[lam_idxs])
        # 2. X-ray binaries (XRB)
        # also have two components:
        #   2.1 high-mass X-ray binaries (HMXB)
        #   2.2 low-mass X-ray binaries (LMXB)
        # Assuming E_cut=100 keV for both components (Wu & Gu 2008)
        # normalized such that L(2-10 keV) = 1.
        self.lumin_lmxb = self.wave**(self.gam_lmxb - 3.) * np.exp(-lam_100keV/self.wave)
        self.lumin_hmxb = self.wave**(self.gam_hmxb - 3.) * np.exp(-lam_100keV/self.wave)
        lam_idxs = (self.wave<=self.lam_2keV) & (self.wave>=self.lam_10keV)
        self.lumin_lmxb /= np.trapz(self.lumin_lmxb[lam_idxs], x=self.wave[lam_idxs])
        self.lumin_hmxb /= np.trapz(self.lumin_hmxb[lam_idxs], x=self.wave[lam_idxs])

        # We compute the unobscured AGN corona X-ray emission
        # The shape is power-law with high-E exp. cutoff
        # with cut-off E=300 keV (Ueda+2014; Aird+2015)
        self.lumin_corona = self.wave**(self.gam - 3.) * np.exp(-lam_300keV/self.wave)
        # Normaliz the SED at 2 keV
        self.lumin_corona /= self.lam_2keV**(self.gam - 3.) * np.exp(-lam_300keV/self.lam_2keV)
        # Calculate total AGN corona X-ray luminosity
        self.l_agn_xray_total = np.trapz(self.lumin_corona, x=self.wave)
        # Calculate 2-10 keV AGN corona X-ray luminosity
        lam_idxs = (self.wave<=self.lam_2keV) & (self.wave>=self.lam_10keV)
        self.l_agn_xray_2to10keV = np.trapz(self.lumin_corona[lam_idxs], x=self.wave[lam_idxs])


    def process(self, sed):
        """Add the X-ray contribution.

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """
        # Stellar info.
        # star formation rate, units: M_sun/yr
        sfr = sed.info['sfh.sfr']
        # stellar mass, units: 1e10 M_sun
        mstar = sed.info['stellar.m_star']/1e10
        # log stellar age, units: Gyr
        logT = np.log10( sed.info['stellar.age_m_star']/1e3 )
        # metallicity, units: none
        Z = sed.info['stellar.metallicity']
        # alpha_ox, added internally
        alpha_ox = self.parameters['alpha_ox']
        # AGN 2500A intrinsic luminosity
        if 'agn.intrin_Lnu_2500A' not in sed.info:
            sed.add_info('agn.intrin_Lnu_2500A', 1., True, unit='W/Hz')
        Lnu_2500A = sed.info['agn.intrin_Lnu_2500A']

        # Add the configuration for X-ray module
        sed.add_module(self.name, self.parameters)
        sed.add_info("xray.gam", self.gam)
        sed.add_info("xray.gam_lmxb", self.gam_lmxb)
        sed.add_info("xray.gam_hmxb", self.gam_hmxb)
        sed.add_info("xray.max_dev_alpha_ox", self.max_dev_alpha_ox)
        sed.add_info("xray.alpha_ox", self.parameters['alpha_ox'])

        # Calculate 0.5-2 keV hot-gas luminosities
        # Mezcua et al. 2018, Eq. 5
        l_hotgas_xray_0p5to2keV = 8.3e31 * sfr
        # Calculate 2-10 keV HMXB luminosities
        # Mezcua et al. 2018, Eq. 1
        l_hmxb_xray_2to10keV = sfr * \
            10**(33.28 - 62.12*Z + 569.44*Z**2 - 1833.80*Z**3 + 1968.33*Z**4)
        # Calculate 2-10 keV LMXB luminosities
        # Mezcua et al. 2018, Eq. 2
        l_lmxb_xray_2to10keV = mstar * \
            10**(33.276 - 1.503*logT - 0.423*logT**2 + 0.425*logT**3 + 0.136*logT**4)

        # Calculate L_lam_2keV from Lnu_2500A
        Lnu_2keV = 10**(alpha_ox/0.3838) * Lnu_2500A
        L_lam_2keV = Lnu_2keV * self.nu_2keV**2/self.c
        # Calculate total AGN corona X-ray luminosity
        l_agn_xray_total = self.l_agn_xray_total * L_lam_2keV
        # Calculate 2-10 keV AGN corona X-ray luminosity
        l_agn_xray_2to10keV = self.l_agn_xray_2to10keV * L_lam_2keV

        # Save the results
        sed.add_info("xray.hotgas_Lx_0p5to2keV", l_hotgas_xray_0p5to2keV, True, unit='W')
        sed.add_info("xray.hmxb_Lx_2to10keV", l_hmxb_xray_2to10keV, True, unit='W')
        sed.add_info("xray.lmxb_Lx_2to10keV", l_lmxb_xray_2to10keV, True, unit='W')
        sed.add_info("xray.agn_Lx_total", l_agn_xray_total, True, unit='W')
        sed.add_info("xray.agn_Lx_2to10keV", l_agn_xray_2to10keV, True, unit='W')
        sed.add_info("xray.agn_Lnu_2keV", Lnu_2keV, True, unit='W/Hz')
        # Add the SED components
        sed.add_contribution('xray.galaxy', self.wave,
                self.lumin_hotgas*l_hotgas_xray_0p5to2keV + \
                self.lumin_lmxb*l_lmxb_xray_2to10keV + \
                self.lumin_hmxb*l_hmxb_xray_2to10keV)
        sed.add_contribution('xray.agn', self.wave, self.lumin_corona * L_lam_2keV)

# SedModule to be returned by get_module
Module = Xray
