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
        ("alpha_ox", (
            "cigale_list()",
            "the power-law slope connecting L_nu at rest-frame 2500 A and 2 keV, "
            "defined as alpha_ox = 0.3838*log(Lnu_2keV/Lnu_2500A).",
            (-1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1)
        )),
        ("max_dev_alpha_ox", (
            "float()",
            "Maximum deviation of alpha_ox from the empirical alpha_ox-Lnu_2500A relation (Just et al. 2007), "
            "i.e. |alpha_ox-alpha_ox(Lnu_2500A)| <= max_dev_alpha_ox. "
            "The alpha_ox-Lnu_2500A relation has a 1-sigma scatter of ~ 0.1. "
            "We assume the relation is measured at a typical AGN viewing angle of 30°.",
            0.2
        )),
        ("angle_coef", (
            "string()",
            "The 1st-order and 2nd-order polynomial coefficients (a1/a2) in X-ray dependence on AGN viewing angle, "
            "i.e. Lx(i)/Lx(0) = a1*cos(i) + a2*cos(i)^2 + 1-a1-a2, where i=0° is face-on and i=90° is edge on. "
            "The viewing angle, i, is defined in the AGN module. "
            "Setting to 0/0 means isotropic AGN X-ray emission.",
            "0/0"
        )),
        ("det_lmxb", (
            "cigale_list()",
            "The deviation from the expected low-mass X-ray binary (LMXB) logLx. "
            "Positive values mean higher logLx from LMXB",
            0.
        )),
        ("det_hmxb", (
            "cigale_list()",
            "The deviation from the expected high-mass X-ray binary (HMXB) logLx. "
            "Positive values mean higher logLx from HMXB",
            0.
        ))
    ])

    def _init_code(self):
        """Build the model for a given set of parameters."""

        self.gam = float(self.parameters["gam"])
        self.angle_coef = str(self.parameters["angle_coef"]).split('/')
        self.a1 = float(self.angle_coef[0])
        self.a2 = float(self.angle_coef[1])
        self.det_lmxb = float(self.parameters["det_lmxb"])
        self.det_hmxb = float(self.parameters["det_hmxb"])
        self.alpha_ox = float(self.parameters["alpha_ox"])
        self.max_dev_alpha_ox = float(self.parameters["max_dev_alpha_ox"])

        # We define various constants necessary to compute the model. For
        # consistency, we define speed of light in units of nm s¯¹
        self.c = cst.c * 1e9
        # Define wavelenght corresponding to some energy in units of nm.
        lam_1keV = self.c*cst.h / (1e3*cst.eV)
        lam_0p5keV = lam_1keV*2
        lam_100keV = lam_1keV*0.01
        lam_300keV = lam_1keV*0.0033333
        lam_2keV = lam_1keV/2
        lam_10keV = lam_1keV*0.1
        lam_8keV = lam_1keV*0.125
        # Define frequency corresponding to 2 keV in units of Hz.
        self.nu_2keV = self.c / lam_2keV

        # We define the wavelength grid for the X-ray emission
        # corresponding to 0.25-1200 keV
        self.wave = np.logspace(-3, 0.7, 1000)

        # X-ray emission from galaxies: 1.hot-gas & 2.X-ray binaries
        # 1.Hot-gas, assuming power-law index gamma=1, E_cut=0.24 keV
        # normalized such that L(0.5-2 keV) = 1
        self.lumin_hotgas = self.wave**-2 * np.exp(-lam_1keV*4.17/self.wave)
        lam_idxs = (self.wave<=lam_0p5keV) & (self.wave>=lam_2keV)
        self.lumin_hotgas *= 1./np.trapz(self.lumin_hotgas[lam_idxs], x=self.wave[lam_idxs])
        # 2. X-ray binaries (XRB)
        # also have two components:
        #   2.1 high-mass X-ray binaries (HMXB), gamma=2
        #   2.2 low-mass X-ray binaries (LMXB), gamma=1.56
        # Assuming E_cut=100 keV for both components (Wu & Gu 2008)
        # normalized such that L(2-10 keV, LMXB)=1. and L(0.5-8 keV, HMXB)=1.
        self.lumin_lmxb = self.wave**(1.56 - 3.) * np.exp(-lam_100keV/self.wave)
        self.lumin_hmxb = self.wave**(2.00 - 3.) * np.exp(-lam_100keV/self.wave)
        lam_idxs = (self.wave<=lam_2keV) & (self.wave>=lam_10keV)
        self.lumin_lmxb /= np.trapz(self.lumin_lmxb[lam_idxs], x=self.wave[lam_idxs])
        lam_idxs = (self.wave<=lam_0p5keV) & (self.wave>=lam_8keV)
        self.lumin_hmxb /= np.trapz(self.lumin_hmxb[lam_idxs], x=self.wave[lam_idxs])

        # We compute the unobscured AGN corona X-ray emission
        # The shape is power-law with high-E exp. cutoff
        # with cut-off E=300 keV (Ueda+2014; Aird+2015)
        self.lumin_corona = self.wave**(self.gam - 3.) * np.exp(-lam_300keV/self.wave)
        # Normaliz the SED at 2 keV
        self.lumin_corona *= 1./(lam_2keV**(self.gam - 3.) * np.exp(-lam_300keV/lam_2keV))
        # Calculate total AGN corona X-ray luminosity
        self.l_agn_xray_total = np.trapz(self.lumin_corona, x=self.wave)
        # Calculate 2-10 keV AGN corona X-ray luminosity
        lam_idxs = (self.wave<=lam_2keV) & (self.wave>=lam_10keV)
        self.l_agn_xray_2to10keV = np.trapz(self.lumin_corona[lam_idxs], x=self.wave[lam_idxs])


    def process(self, sed):
        """Add the X-ray contribution.

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """
        # Stellar info.
        # star formation rate, units: M_sun/yr
        sfr = sed.info['sfh.sfr100Myrs']
        # stellar mass, units: 1e10 M_sun
        if 'stellar.m_star' in sed.info:
            mstar = sed.info['stellar.m_star']*1e-10
        elif 'stellar.mass_total' in sed.info:
            mstar = sed.info['stellar.mass_total']*1e-10
        else:
            raise Exception("a stellar module is required")
        # log stellar age, units: Gyr
        logT = np.log10( sed.info['stellar.age_m_star']*1e-3 )
        # log metallicity, units: none
        logZ = np.log10(sed.info['stellar.metallicity'])
        # Get AGN viewing angle and 2500A intrinsic luminosity (at 30 deg)
        if 'agn.i' in sed.info:
            # SKIRTOR model
            cosi = np.cos(sed.info['agn.i']*np.pi/180)
        elif 'agn.psy' in sed.info:
            # Fritz model
            cosi = np.sin(sed.info['agn.psy']*np.pi/180)
        else:
            raise Exception("An AGN module (skirtor2016 or fritz2006) is required")
        Lnu_2500A = sed.info['agn.intrin_Lnu_2500A_30deg']

        # Add the configuration for X-ray module
        sed.add_module(self.name, self.parameters)
        sed.add_info("xray.gam", self.gam)
        sed.add_info("xray.det_lmxb", self.det_lmxb)
        sed.add_info("xray.det_hmxb", self.det_hmxb)
        sed.add_info("xray.alpha_ox", self.alpha_ox)
        sed.add_info("xray.max_dev_alpha_ox", self.max_dev_alpha_ox)

        # Calculate 0.5-2 keV hot-gas luminosities
        l_hotgas_xray_0p5to2keV = 8.3e31 * sfr
        # Calculate 2-10 keV HMXB luminosities
        l_hmxb_xray_0p5to8keV = sfr * \
            10**(30.83 - 1.33*logZ - 0.17*logZ**2 \
                 + self.det_hmxb)
        # Calculate 2-10 keV LMXB luminosities
        l_lmxb_xray_2to10keV = mstar * \
            10**(33.276 - 1.503*logT - 0.423*logT**2 + 0.425*logT**3 + 0.136*logT**4 \
                 + self.det_lmxb)

        # Calculate L_lam_2keV from Lnu_2500A (both at 30 deg)
        Lnu_2keV = 10**(self.alpha_ox/0.3838) * Lnu_2500A
        L_lam_2keV = Lnu_2keV * self.nu_2keV**2/self.c
        # Calculate total AGN corona X-ray luminosity at theta deg
        scl_fac = (self.a1*cosi + self.a2*cosi**2 + 1-self.a1-self.a2) /\
                  (1 - 0.13397*self.a1 - 0.25*self.a2) * L_lam_2keV
        l_agn_xray_total = self.l_agn_xray_total * scl_fac
        # Calculate 2-10 keV AGN corona X-ray luminosity at theta deg
        l_agn_xray_2to10keV = self.l_agn_xray_2to10keV * scl_fac

        # Save the results
        sed.add_info("xray.hotgas_Lx_0p5to2keV", l_hotgas_xray_0p5to2keV, True, unit='W')
        sed.add_info("xray.hmxb_Lx_0p5to8keV", l_hmxb_xray_0p5to8keV, True, unit='W')
        sed.add_info("xray.lmxb_Lx_2to10keV", l_lmxb_xray_2to10keV, True, unit='W')
        sed.add_info("xray.agn_Lx_total", l_agn_xray_total, True, unit='W')
        sed.add_info("xray.agn_Lx_2to10keV", l_agn_xray_2to10keV, True, unit='W')
        sed.add_info("xray.agn_Lnu_2keV_30deg", Lnu_2keV, True, unit='W/Hz')
        # Add the SED components
        sed.add_contribution('xray.galaxy', self.wave,
                self.lumin_hotgas*l_hotgas_xray_0p5to2keV + \
                self.lumin_lmxb*l_lmxb_xray_2to10keV + \
                self.lumin_hmxb*l_hmxb_xray_0p5to8keV)
        sed.add_contribution('xray.agn', self.wave, self.lumin_corona * scl_fac)

# SedModule to be returned by get_module
Module = Xray
