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
            "Possible values are: 3, 5, 7, 9, and 11.",
            7
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
            "Half-opening angle of the dust-free cone is 90°-oa. "
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
            "inclination, i.e. viewing angle, position of the instrument "
            "w.r.t. the AGN axis. i=[0, 90°-oa): face-on, type 1 view; "
            "i=[90°-oa, 90°]: edge-on, type 2 view. "
            "Possible values are: 0, 10, 20, 30, 40, 50, 60, 70, 80, and 90.",
            30
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
        self.t = int(self.parameters["t"])
        self.pl = float(self.parameters["pl"])
        self.q = float(self.parameters["q"])
        self.oa = int(self.parameters["oa"])
        self.R = int(self.parameters["R"])
        self.Mcl = float(self.parameters["Mcl"])
        self.i = int(self.parameters["i"])
        self.fracAGN = float(self.parameters["fracAGN"])
        self.law = int(self.parameters["law"])
        self.EBV = float(self.parameters["EBV"])
        self.temperature = float(self.parameters["temperature"])
        self.emissivity = float(self.parameters["emissivity"])

        with Database() as base:
            self.SKIRTOR2016 = base.get_skirtor2016(self.t, self.pl, self.q,
                                                    self.oa, self.R, self.Mcl,
                                                    self.i)

        # Apply polar-dust obscuration
        # We define various constants necessary to compute the model
        self.c = cst.c * 1e9
        lambda_0 = 200e3
        # Calculate the extinction
        k_lam = k_ext(self.SKIRTOR2016.wave, self.law)
        1.39*(self.SKIRTOR2016.wave/1e3)**-1.2
        A_lam = k_lam * self.EBV
        # The extinction factor, flux_1/flux_0
        ext_fac = 10**(A_lam/-2.5)
        # Calculate the new AGN SED shape after extinction
        if self.i <= (90-self.oa):
            # The direct and scattered components (line-of-sight) are extincted for type-1 AGN
            disk_new = self.SKIRTOR2016.disk * ext_fac
        else:
            # Keep the direct and scatter components for type-2
            disk_new = self.SKIRTOR2016.disk
        # Calculate the total extincted luminosity averaged over all directions
        sin_oa = np.sin( self.oa*np.pi/180 )
        l_ext = np.trapz(self.SKIRTOR2016.intrin_disk * (1-ext_fac),
                         x=self.SKIRTOR2016.wave) * \
                        (0.4931 - 0.2113*sin_oa**2 - 0.2818*sin_oa**3)
        # Casey (2012) modified black body model
        conv = self.c / (self.SKIRTOR2016.wave * self.SKIRTOR2016.wave)
        # To avoid inf occurance in exponential, set blackbody=0 when h*c/lam*k*T is large
        hc_lkt = cst.h * self.c / (self.SKIRTOR2016.wave * cst.k * self.temperature)
        non_0_idxs = hc_lkt<100
        # Generate the blackbody
        blackbody = np.zeros(len(self.SKIRTOR2016.wave))
        blackbody[non_0_idxs] = conv[non_0_idxs] * \
            (1. - np.exp(-(lambda_0 / self.SKIRTOR2016.wave[non_0_idxs])** self.emissivity)) * \
            (self.c / self.SKIRTOR2016.wave[non_0_idxs]) ** 3. / \
            (np.exp(hc_lkt[non_0_idxs]) - 1.)
        blackbody *= l_ext / np.trapz(blackbody, x=self.SKIRTOR2016.wave)
        # Add the black body to dust thermal emission
        dust_new = self.SKIRTOR2016.dust + blackbody
        # Normalize direct, scatter, and thermal components
        norm = np.trapz(dust_new, x=self.SKIRTOR2016.wave)
        dust_new /= norm
        disk_new /= norm
        # Update SKIRTOR model SED
        self.SKIRTOR2016.dust = dust_new
        self.SKIRTOR2016.disk = disk_new
        self.SKIRTOR2016.intrin_disk /= norm

        # Integrate AGN luminosity for different components
        self.lumin_disk = np.trapz(self.SKIRTOR2016.disk, x=self.SKIRTOR2016.wave)
        # Intrinsic (de-reddened) AGN luminosity from the central source at theta=30 deg
        self.lumin_intrin_disk = np.trapz(self.SKIRTOR2016.intrin_disk,
                                          x=self.SKIRTOR2016.wave)
        # Calculate L_lam(2500A) at theta=30 deg
        self.l_agn_2500A = np.interp(250, self.SKIRTOR2016.wave, self.SKIRTOR2016.intrin_disk)
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
        sed.add_info('agn.t', self.t)
        sed.add_info('agn.pl', self.pl)
        sed.add_info('agn.q', self.q)
        sed.add_info('agn.oa', self.oa, unit='deg')
        sed.add_info('agn.R', self.R)
        sed.add_info('agn.Mcl', self.Mcl)
        sed.add_info('agn.i', self.i, unit='deg')
        sed.add_info('agn.fracAGN', self.fracAGN)
        sed.add_info('agn.law', self.law)
        sed.add_info('agn.EBV', self.EBV)
        sed.add_info('agn.temperature', self.temperature)
        sed.add_info('agn.emissivity', self.emissivity)

        # Compute the AGN luminosity
        if self.fracAGN < 1.:
            agn_power = luminosity * (1./(1.-self.fracAGN) - 1.)
            lumin_dust = agn_power
            lumin_disk = agn_power * self.lumin_disk
            # power_accretion means the intrinsic disk luminosity
            # integrated over 4pi solid angles
            # The factor (0.493) comes from the fact that lumin_intrin_disk
            # is calculated at viewing angle = 30 deg
            power_accretion = agn_power * self.lumin_intrin_disk * 0.493
            lumin = lumin_dust + lumin_disk
            l_agn_2500A = agn_power * self.l_agn_2500A
        else:
            raise Exception("AGN fraction is exactly 1. Behaviour "
                            "undefined.")

        sed.add_info('agn.dust_luminosity', lumin_dust, True, unit='W')
        sed.add_info('agn.disk_luminosity', lumin_disk, True, unit='W')
        sed.add_info('agn.luminosity', lumin, True, unit='W')
        sed.add_info('agn.accretion_power', power_accretion, True)
        sed.add_info('agn.intrin_Lnu_2500A', l_agn_2500A, True)

        sed.add_contribution('agn.SKIRTOR2016_dust', self.SKIRTOR2016.wave,
                             agn_power * self.SKIRTOR2016.dust)
        sed.add_contribution('agn.SKIRTOR2016_disk', self.SKIRTOR2016.wave,
                             agn_power * self.SKIRTOR2016.disk)


# SedModule to be returned by get_module
Module = SKIRTOR2016
