# -*- coding: utf-8 -*-
# Copyright (C) 2013, 2014 Department of Physics, University of Crete
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Laure Ciesla

"""
Fritz et al. (2006) AGN dust torus emission module
==================================================

This module implements the Fritz et al. (2006) models.

"""
from collections import OrderedDict

import numpy as np

from pcigale.data import Database
from . import SedModule

import scipy.constants as cst

def k_ext(wavelength, ext_law):
    """
    Compute k(λ)=A(λ)/E(B-V) for a specified extinction law

    Parameters
    ----------
    wavelength: array of floats
        Wavelength grid in nm.
    ext_law: the extinction law
             0=SMC, 1=Calzetti2000, 2=Gaskell2004

    Returns
    -------
    a numpy array of floats

    """
    if ext_law==0:
        # SMC, from Bongiorno+2012
        return 1.39*(wavelength/1e3)**-1.2
    elif ext_law==1:
        # Calzetti2000, from dustatt_calzleit.py
        result = np.zeros(len(wavelength))
        # Attenuation between 120 nm and 630 nm
        mask = (wavelength < 630)
        result[mask] = 2.659 * (-2.156 + 1.509e3 / wavelength[mask] -
                                0.198e6 / wavelength[mask] ** 2 +
                                0.011e9 / wavelength[mask] ** 3) + 4.05
        # Attenuation between 630 nm and 2200 nm
        mask = (wavelength >= 630)
        result[mask] = 2.659 * (-1.857 + 1.040e3 / wavelength[mask]) + 4.05
        return result
    elif ext_law==2:
        # Gaskell+2004, from the appendix of that paper
        x = 1 / (wavelength/1e3)
        Alam_Av = np.zeros(len(wavelength))
        # Attenuation for x = 1.6 -- 3.69
        mask = (x < 3.69)
        Alam_Av[mask] = -0.8175 + 1.5848*x[mask] - 0.3774*x[mask]**2 + 0.0296*x[mask]**3
        # Attenuation for x = 3.69 -- 8
        mask = (x >= 3.69)
        Alam_Av[mask] = 1.3468 + 0.0087*x[mask]
        # Set negative values to zero
        Alam_Av[Alam_Av<0] = 0
        # Convert A(λ)/A(V) to A(λ)/E(B-V)
        # assuming A(B)/A(V) = 1.182 (Table 3 of Gaskell+2004)
        return Alam_Av/0.182
    else:
        raise KeyError("Extinction law is different from the expected ones")


class Fritz2006(SedModule):
    """Fritz et al. (2006) AGN dust torus emission

    The AGN emission is computed from the library of Fritz et al. (2006) from
    which all of the models are available. They take into account two emission
    components linked to the AGN. The first one is the isotropic emission of
    the central source, which is assumed to be point-like. This emission is a
    composition of power laws with variable indices, in the wavelength range of
    0.001-20 microns. The second one is the thermal and scattering dust torus
    emission. The conservation of the energy is always verified within 1% for
    typical solutions, and up to 10% in the case of very high optical depth and
    non-constant dust density. We refer the reader to Fritz et al. (2006) for
    more information on the library.

    The relative normalization of these components is handled through a
    parameter which is the fraction of the total IR luminosity due to the AGN
    so that: L_AGN = fracAGN * L_IRTOT, where L_AGN is the AGN luminosity,
    fracAGN is the contribution of the AGN to the total IR luminosity
    (L_IRTOT), i.e. L_Starburst+L_AGN.

    """

    parameter_list = OrderedDict([
        ('r_ratio', (
            "cigale_list(options=10. & 30. & 60. & 100. & 150.)",
            "Ratio of the maximum to minimum radii of the dust torus. "
            "Possible values are: 10, 30, 60, 100, 150.",
            60.
        )),
        ('tau', (
            "cigale_list(options=0.1 & 0.3 & 0.6 & 1.0 & 2.0 & 3.0 & 6.0 & "
            "10.0)",
            "Optical depth at 9.7 microns. "
            "Possible values are: 0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0.",
            1.0
        )),
        ('beta', (
            "cigale_list(options=-1.00 & -0.75 & -0.50 & -0.25 & 0.00)",
            "Beta. Possible values are: -1.00, -0.75, -0.50, -0.25, 0.00.",
            -0.50
        )),
        ('gamma', (
            'cigale_list(options=0.0 & 2.0 & 4.0 & 6.0)',
            "Gamma. Possible values are: 0.0, 2.0, 4.0, 6.0.",
            4.0
        )),
        ('opening_angle', (
            'cigale_list(options=60. & 100. & 140.)',
            "Full opening angle of the dust torus (Fig 1 of Fritz 2006). "
            "Possible values are: 60., 100., 140.",
            100.
        )),
        ('psy', (
            'cigale_list(options=0.001 & 10.100 & 20.100 & 30.100 & 40.100 & '
            '50.100 & 60.100 & 70.100 & 80.100 & 89.990)',
            "Angle between equatorial axis and line of sight. "
            "Psy = 90◦ for type 1 and Psy = 0° for type 2. Possible values "
            "are: 0.001, 10.100, 20.100, 30.100, 40.100, 50.100, 60.100, "
            "70.100, 80.100, 89.990.",
            50.100
        )),
        ('fracAGN', (
            'cigale_list(minvalue=0., maxvalue=1.)',
            "AGN fraction.",
            0.1
        )),
        ('law', (
            'cigale_list(dtype=int, options=0 & 1 & 2)',
            "The extinction law of polar dust: "
            "0 (SMC), 1 (Calzetti 2000), or 2 (Gaskell et al. 2004)",
            0
        )),
        ('EBV', (
            'cigale_list(minvalue=0.)',
            "E(B-V) for extinction in polar direction",
            0.1
        )),
        ('temperature', (
            'cigale_list(minvalue=0.)',
            "Temperature of the polar dust in K",
            100.
        )),
        ("emissivity", (
            "cigale_list(minvalue=0.)",
            "Emissivity index of the polar dust",
            1.6
        ))
   ])

    def _init_code(self):
        """Get the template set out of the database"""
        self.r_ratio = float(self.parameters["r_ratio"])
        self.tau = float(self.parameters["tau"])
        self.beta = float(self.parameters["beta"])
        self.gamma = float(self.parameters["gamma"])
        self.opening_angle = (180. - self.parameters["opening_angle"]) / 2.
        self.psy = float(self.parameters["psy"])
        self.fracAGN = float(self.parameters["fracAGN"])
        self.law = int(self.parameters["law"])
        self.EBV = float(self.parameters["EBV"])
        self.temperature = float(self.parameters["temperature"])
        self.emissivity = float(self.parameters["emissivity"])

        with Database() as base:
            self.fritz2006 = base.get_fritz2006(self.r_ratio, self.tau,
                                                self.beta, self.gamma,
                                                self.opening_angle, self.psy)
        self.l_agn_scatt = np.trapz(self.fritz2006.lumin_scatt,
                                    x=self.fritz2006.wave)
        self.l_agn_agn = np.trapz(self.fritz2006.lumin_agn,
                                  x=self.fritz2006.wave)

        # Apply wavelength cut to avoid X-ray wavelength
        lam_cut = 10**0.9
        lam_idxs = self.fritz2006.wave>=lam_cut
        # Calculate the re-normalization factor to keep energy conservation
        norm_fac = np.trapz(self.fritz2006.lumin_intrin_agn, x=self.fritz2006.wave) /\
            np.trapz(self.fritz2006.lumin_intrin_agn[lam_idxs], x=self.fritz2006.wave[lam_idxs])
        # Perform the cut
        self.fritz2006.wave = self.fritz2006.wave[lam_idxs]
        self.fritz2006.lumin_agn = self.fritz2006.lumin_agn[lam_idxs]*norm_fac
        self.fritz2006.lumin_scatt = self.fritz2006.lumin_scatt[lam_idxs]*norm_fac
        self.fritz2006.lumin_therm  = self.fritz2006.lumin_therm[lam_idxs]
        self.fritz2006.lumin_intrin_agn = self.fritz2006.lumin_intrin_agn[lam_idxs]*norm_fac

        # Apply polar-dust obscuration
        # We define various constants necessary to compute the model
        self.c = cst.c * 1e9
        lambda_0 = 200e3
        # Calculate the extinction (SMC)
        # The analytical formula is from Bongiorno+2012
        k_lam = k_ext(self.fritz2006.wave, self.law)
        A_lam = k_lam * self.EBV
        # The extinction factor, flux_1/flux_0
        ext_fac = 10**(A_lam/-2.5)
        # Calculate the new AGN SED shape after extinction
        if self.psy > (90-self.opening_angle):
            # The direct and scattered components (line-of-sight) are extincted for type-1 AGN
            lumin_agn_new = self.fritz2006.lumin_agn * ext_fac
            lumin_scatt_new = self.fritz2006.lumin_scatt * ext_fac
        else:
            # Keep the direct and scatter components for type-2
            lumin_agn_new = self.fritz2006.lumin_agn
            lumin_scatt_new = self.fritz2006.lumin_scatt
        # Calculate the total extincted luminosity averaged over all directions
        # note that self.opening_angle is different from Fritz's definiting!
        l_ext = np.trapz(self.fritz2006.lumin_intrin_agn * (1-ext_fac),
                         x=self.fritz2006.wave) * \
                        (1 - np.cos( np.deg2rad(self.opening_angle) ))
        # Casey (2012) modified black body model
        conv = self.c / (self.fritz2006.wave * self.fritz2006.wave)
        # To avoid inf occurance in exponential, set blackbody=0 when h*c/lam*k*T is large
        hc_lkt = cst.h * self.c / (self.fritz2006.wave * cst.k * self.temperature)
        non_0_idxs = hc_lkt<100
        # Generate the blackbody
        lumin_blackbody = np.zeros(len(self.fritz2006.wave))
        lumin_blackbody[non_0_idxs] = conv[non_0_idxs] * \
            (1. - np.exp(-(lambda_0 / self.fritz2006.wave[non_0_idxs])** self.emissivity)) * \
            (self.c / self.fritz2006.wave[non_0_idxs]) ** 3. / \
            (np.exp(hc_lkt[non_0_idxs]) - 1.)
        lumin_blackbody *= l_ext / np.trapz(lumin_blackbody, x=self.fritz2006.wave)
        # Add the black body to dust thermal emission
        lumin_therm_new = self.fritz2006.lumin_therm + lumin_blackbody
        # Normalize direct, scatter, and thermal components
        norm = np.trapz(lumin_therm_new, x=self.fritz2006.wave)
        lumin_therm_new /= norm
        lumin_scatt_new /= norm
        lumin_agn_new /= norm
        # Update fritz2006 lumin
        self.fritz2006.lumin_therm = lumin_therm_new
        self.fritz2006.lumin_scatt = lumin_scatt_new
        self.fritz2006.lumin_agn = lumin_agn_new
        self.fritz2006.lumin_intrin_agn /= norm

        # Integrate AGN luminosity for different components
        self.l_agn_scatt = np.trapz(self.fritz2006.lumin_scatt, x=self.fritz2006.wave)
        self.l_agn_agn = np.trapz(self.fritz2006.lumin_agn, x=self.fritz2006.wave)
        # Intrinsic (de-reddened) AGN luminosity from the central source
        self.l_agn_intrin_agn = np.trapz(self.fritz2006.lumin_intrin_agn, x=self.fritz2006.wave)
        # Calculate L_lam(2500A)
        self.l_agn_2500A = np.interp(250, self.fritz2006.wave, self.fritz2006.lumin_intrin_agn)
        # Convert L_lam to L_nu
        self.l_agn_2500A *= 250**2/self.c

        # Apply wavelength cut to avoid X-ray wavelength
        lam_cut = 10**0.9
        lam_idxs = self.fritz2006.wave>=lam_cut
        # Calculate the re-normalization factor to keep energy conservation
        norm_fac = np.trapz(self.fritz2006.lumin_intrin_agn, x=self.fritz2006.wave) /\
            np.trapz(self.fritz2006.lumin_intrin_agn[lam_idxs], x=self.fritz2006.wave[lam_idxs])
        # Perform the cut
        self.fritz2006.wave = self.fritz2006.wave[lam_idxs]
        self.fritz2006.lumin_agn = self.fritz2006.lumin_agn[lam_idxs]*norm_fac
        self.fritz2006.lumin_scatt = self.fritz2006.lumin_scatt[lam_idxs]*norm_fac
        self.fritz2006.lumin_therm  = self.fritz2006.lumin_therm[lam_idxs]
        self.fritz2006.lumin_intrin_agn = self.fritz2006.lumin_intrin_agn[lam_idxs]*norm_fac

        # Apply polar-dust obscuration
        # We define various constants necessary to compute the model
        self.c = cst.c * 1e9
        lambda_0 = 200e3
        # Calculate the extinction (SMC)
        # The analytical formula is from Bongiorno+2012
        k_lam = k_ext(self.fritz2006.wave, self.law)
        A_lam = k_lam * self.EBV
        # The extinction factor, flux_1/flux_0
        ext_fac = 10**(A_lam/-2.5)
        # Calculate the new AGN SED shape after extinction
        if self.psy > (90-self.opening_angle):
            # The direct and scattered components (line-of-sight) are extincted for type-1 AGN
            lumin_agn_new = self.fritz2006.lumin_agn * ext_fac
            lumin_scatt_new = self.fritz2006.lumin_scatt * ext_fac
        else:
            # Keep the direct and scatter components for type-2
            lumin_agn_new = self.fritz2006.lumin_agn
            lumin_scatt_new = self.fritz2006.lumin_scatt
        # Calculate the total extincted luminosity averaged over all directions
        # note that self.opening_angle is different from Fritz's definiting!
        l_ext = np.trapz(self.fritz2006.lumin_intrin_agn * (1-ext_fac),
                         x=self.fritz2006.wave) * \
                        (1 - np.cos( np.deg2rad(self.opening_angle) ))
        # Casey (2012) modified black body model
        conv = self.c / (self.fritz2006.wave * self.fritz2006.wave)
        # To avoid inf occurance in exponential, set blackbody=0 when h*c/lam*k*T is large
        hc_lkt = cst.h * self.c / (self.fritz2006.wave * cst.k * self.temperature)
        non_0_idxs = hc_lkt<100
        # Generate the blackbody
        lumin_blackbody = np.zeros(len(self.fritz2006.wave))
        lumin_blackbody[non_0_idxs] = conv[non_0_idxs] * \
            (1. - np.exp(-(lambda_0 / self.fritz2006.wave[non_0_idxs])** self.emissivity)) * \
            (self.c / self.fritz2006.wave[non_0_idxs]) ** 3. / \
            (np.exp(hc_lkt[non_0_idxs]) - 1.)
        lumin_blackbody *= l_ext / np.trapz(lumin_blackbody, x=self.fritz2006.wave)
        # Add the black body to dust thermal emission
        lumin_therm_new = self.fritz2006.lumin_therm + lumin_blackbody
        # Normalize direct, scatter, and thermal components
        norm = np.trapz(lumin_therm_new, x=self.fritz2006.wave)
        lumin_therm_new /= norm
        lumin_scatt_new /= norm
        lumin_agn_new /= norm
        # Update fritz2006 lumin
        self.fritz2006.lumin_therm = lumin_therm_new
        self.fritz2006.lumin_scatt = lumin_scatt_new
        self.fritz2006.lumin_agn = lumin_agn_new
        self.fritz2006.lumin_intrin_agn /= norm

        # Integrate AGN luminosity for different components
        self.l_agn_scatt = np.trapz(self.fritz2006.lumin_scatt, x=self.fritz2006.wave)
        self.l_agn_agn = np.trapz(self.fritz2006.lumin_agn, x=self.fritz2006.wave)
        # Intrinsic (de-reddened) AGN luminosity from the central source
        self.l_agn_intrin_agn = np.trapz(self.fritz2006.lumin_intrin_agn, x=self.fritz2006.wave)
        # Calculate L_lam(2500A)
        self.l_agn_2500A = np.interp(250, self.fritz2006.wave, self.fritz2006.lumin_intrin_agn)
        # Convert L_lam to L_nu
        self.l_agn_2500A *= 250**2/self.c

    def process(self, sed):
        """Add the IR re-emission contributions

        Parameters
        ----------
        sed: pcigale.sed.SED object
        parameters: dictionary containing the parameters

        """

        if 'dust.luminosity' not in sed.info:
            sed.add_info('dust.luminosity', 1., True, unit='W')
        luminosity = sed.info['dust.luminosity']

        sed.add_module(self.name, self.parameters)
        sed.add_info('agn.r_ratio', self.r_ratio)
        sed.add_info('agn.tau', self.tau)
        sed.add_info('agn.beta', self.beta)
        sed.add_info('agn.gamma', self.gamma)
        sed.add_info('agn.opening_angle', self.parameters["opening_angle"],
                     unit='deg')
        sed.add_info('agn.psy', self.psy, unit='deg')
        sed.add_info('agn.fracAGN', self.fracAGN)
        sed.add_info('agn.law', self.law)
        sed.add_info('agn.EBV', self.EBV)
        sed.add_info('agn.temperature', self.temperature, unit='K')
        sed.add_info('agn.emissivity', self.emissivity)

        # Compute the AGN luminosity
        if self.fracAGN < 1.:
            agn_power = luminosity * (1./(1.-self.fracAGN) - 1.)
            l_agn_therm = agn_power
            l_agn_scatt = agn_power * self.l_agn_scatt
            l_agn_agn = agn_power * self.l_agn_agn
            l_agn_total = l_agn_therm + l_agn_scatt + l_agn_agn
            l_agn_intrin_agn = agn_power * self.l_agn_intrin_agn
            l_agn_2500A = agn_power * self.l_agn_2500A
        else:
            raise Exception("AGN fraction is exactly 1. Behaviour "
                            "undefined.")

        sed.add_info('agn.therm_luminosity', l_agn_therm, True, unit='W')
        sed.add_info('agn.scatt_luminosity', l_agn_scatt, True, unit='W')
        sed.add_info('agn.disk_luminosity', l_agn_agn, True, unit='W')
        sed.add_info('agn.luminosity', l_agn_total, True, unit='W')
        sed.add_info('agn.accretion_power', l_agn_intrin_agn, True, unit='W')
        sed.add_info('agn.intrin_Lnu_2500A', l_agn_2500A, True, unit='W/Hz')

        sed.add_contribution('agn.fritz2006_therm', self.fritz2006.wave,
                             agn_power * self.fritz2006.lumin_therm)
        sed.add_contribution('agn.fritz2006_scatt', self.fritz2006.wave,
                             agn_power * self.fritz2006.lumin_scatt)
        sed.add_contribution('agn.fritz2006_agn', self.fritz2006.wave,
                             agn_power * self.fritz2006.lumin_agn)


# SedModule to be returned by get_module
Module = Fritz2006
