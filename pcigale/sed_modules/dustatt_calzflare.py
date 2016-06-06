# -*- coding: utf-8 -*-
# Copyright (C) 2016 Laboratoire d'Astrophysique de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt

"""
Flare Calzetti et al. (2000) and Leitherer et al. (2002) attenuation module
===========================================================================

This module implements the Calzetti et al. (2000) and  Leitherer et al. (2002)
attenuation formulae, adding an UV-bump and a power law.

"""

from collections import OrderedDict

import numpy as np

from . import SedModule


def k_calzetti2000(wavelength):
    """Compute the Calzetti et al. (2000) A(λ)/E(B-V)∗

    Given a wavelength grid, this function computes the selective attenuation
    A(λ)/E(B-V)∗ using the formula from Calzetti at al. (2000). This formula
    is given for wavelengths between 120 nm and 2200 nm, but this function
    makes the computation outside.

    Parameters
    ----------
    wavelength: array of floats
        Wavelength grid in nm.

    Returns
    -------
    a numpy array of floats

    """
    wavelength = np.array(wavelength)
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


def k_leitherer2002(wavelength):
    """Compute the Leitherer et al. (2002) A(λ)/E(B-V)∗

    Given a wavelength grid, this function computes the selective attenuation
    A(λ)/E(B-V)∗ using the formula from Leitherer at al. (2002). This formula
    is given for wavelengths between 91.2 nm and 180 nm, but this function
    makes the computation outside.

    Parameters
    ----------
    wavelength: array of floats
        Wavelength grid in nm.

    Returns
    -------
    a numpy array of floats

    """
    wavelength = np.array(wavelength)
    result = (5.472 + 0.671e3 / wavelength -
              9.218e3 / wavelength ** 2 +
              2.620e6 / wavelength ** 3)

    return result


def uv_bump(wavelength, central_wave, gamma, ebump):
    """Compute the Lorentzian-like Drude profile.

    Parameters
    ----------
    wavelength: array of floats
        Wavelength grid in nm.
    central_wave: float
        Central wavelength of the bump in nm.
    gamma: float
        Width (FWHM) of the bump in nm.
    ebump: float
        Amplitude of the bump.

    Returns
    -------
    a numpy array of floats

    """
    return (ebump * wavelength ** 2 * gamma ** 2 /
            ((wavelength ** 2 - central_wave ** 2) ** 2 +
             wavelength ** 2 * gamma ** 2))


def power_law(wavelength, delta):
    """Power law 'centered' on 550 nm..

    Parameters
    ----------
    wavelength: array of floats
        The wavelength grid in nm.
    delta: float
        The slope of the power law.

    Returns
    -------
    array of floats

    """
    return (wavelength / 550) ** delta


def a_vs_ebv(wavelength, bump_wave, bump_width, bump_ampl, power_slope):
    """Compute the complete attenuation curve A(λ)/E(B-V)*

    The Leitherer et al. (2002) formula is used between 91.2 nm and 150 nm, and
    the Calzetti et al. (2000) formula is used after 150 (we do an
    extrapolation after 2200 nm). When the attenuation becomes negative, it is
    kept to 0. This continuum is multiplied by the power law and then the UV
    bump is added.

    Parameters
    ----------
    wavelength: array of floats
        The wavelength grid (in nm) to compute the attenuation curve on.
    bump_wave: float
        Central wavelength (in nm) of the UV bump.
    bump_width: float
        Width (FWHM, in nm) of the UV bump.
    bump_ampl: float
        Amplitude of the UV bump.
    power_slope: float
        Slope of the power law.

    Returns
    -------
    attenuation: array of floats
        The A(λ)/E(B-V)* attenuation at each wavelength of the grid.

    """
    attenuation = np.zeros(len(wavelength))

    # Leitherer et al.
    mask = (wavelength > 91.2) & (wavelength < 150)
    attenuation[mask] = k_leitherer2002(wavelength[mask])
    # Calzetti et al.
    mask = (wavelength >= 150)
    attenuation[mask] = k_calzetti2000(wavelength[mask])
    # We set attenuation to 0 where it becomes negative
    mask = (attenuation < 0)
    attenuation[mask] = 0
    # Power law
    attenuation *= power_law(wavelength, power_slope)
    # UV bump
    attenuation += uv_bump(wavelength, bump_wave, bump_width, bump_ampl)

    # As the powerlaw slope changes E(B-V), we correct this so that the curve
    # always has the same E(B-V) as the starburst curve. This ensures that the
    # E(B-V) requested by the user is the actual E(B-V) of the curve.
    wl_BV = np.array([440., 550.])
    EBV_calz = ((k_calzetti2000(wl_BV) * power_law(wl_BV, 0.)) +
                uv_bump(wl_BV, bump_wave, bump_width, bump_ampl))
    EBV = ((k_calzetti2000(wl_BV) * power_law(wl_BV, power_slope)) +
           uv_bump(wl_BV, bump_wave, bump_width, bump_ampl))
    attenuation *= (EBV_calz[1]-EBV_calz[0]) / (EBV[1]-EBV[0])

    return attenuation


class CalzFlare(SedModule):
    """Flare Calzetti + Leitherer attenuation module

    """

    parameter_list = OrderedDict([
        #("E_BVs_old_factor", (
        #    "cigale_list(minvalue=0., maxvalue=1.)",
        #    "Reduction factor for the E(B-V)* of the old population compared "
        #    "to the young one (<1).",
        #    1.
        #)),
        ("uv_bump_wavelength", (
            "cigale_list(minvalue=0.)",
            "Central wavelength of the UV bump in nm.",
            217.5
        )),
        ("uv_bump_width", (
            "cigale_list()",
            "Width (FWHM) of the UV bump in nm.",
            35.
        )),
        ("uv_bump_amplitude", (
            "cigale_list(minvalue=0.)",
            "Amplitude of the UV bump. For the Milky Way: 3.",
            0.
        )),
        ("powerlaw_slope", (
            "cigale_list()",
            "Slope delta of the power law modifying the attenuation curve.",
            0.
        )),
        ("filters", (
            "string()",
            "Filters for which the attenuation will be computed and added to "
            "the SED information dictionary. You can give several filter "
            "names separated by a & (don't use commas).",
            "B_B90 & V_B90 & FUV"
        ))
    ])

    def _init_code(self):
        """Get the filters from the database"""
        #self.ebvs_old_factor = float(self.parameters["E_BVs_old_factor"])
        self.ebvs_old_factor = 0.44
        self.uv_bump_wavelength = float(self.parameters["uv_bump_wavelength"])
        self.uv_bump_width = float(self.parameters["uv_bump_width"])
        self.uv_bump_amplitude = float(self.parameters["uv_bump_amplitude"])
        self.powerlaw_slope = float(self.parameters["powerlaw_slope"])
        self.filter_list = [item.strip() for item in
                            self.parameters["filters"].split("&")]

    def process(self, sed):
        """Add the CCM dust attenuation to the SED.

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """
        redshift = sed.info['universe.redshift']
        wavelength = sed.wavelength_grid

        # Computation of the E_BVs_young from the mass of the galaxy.
        ebvs = {}
        # BC03 SSPS used
        if 'stellar.m_star' in sed.info:
            m_star = sed.info['stellar.m_star']
            m_star_old = sed.info['stellar.m_star_old']
            m_star_young = sed.info['stellar.m_star_young']
        # Maraston (2005) SSPs used
        elif 'stellar.mass_total' in sed.info:
            m_star = sed.info['stellar.mass_total']
            m_star_old = sed.info['stellar.m_star_old']
            m_star_young = sed.info['stellar.m_star_young']
        sfr10Myrs = sed.info['sfh.sfr10Myrs']

        # Adapted from Pannella et al. (2015, ApJ 807, 141) for 1.2 < z < 4.0
        #a_fuv = max(0., 1.6 * np.log10(m_star) - 13.5)
        if np.log10(m_star) >= 7.0:
            #a_fuv = -8.1e-3*(np.log10(m_star))**3 + 0.38*(np.log10(m_star))**2 - \
            #         3.94*np.log10(m_star) + 12.0
            #a_fuv = 8e-3*(np.log10(m_star))**3) - 0.062*(np.log10(m_star))**2 + \
            #         0.067*np.log10(m_star)
            #a_fuv = 2e-3*(np.log10(m_star))**3) - 0.01*(np.log10(m_star))**2 + \
            #         0.03*np.log10(m_star)

            #(0.05 + 0.25*exp(-pow((6.-2.0)/1.5, 2)) + 0.03*exp(-pow((6.-4.5)/2.0, 2))) * pow(logMstar-7, 2)
            #A_fuv_mf_old = (0.05 + 0.25*np.exp(-pow((z-2.0)/1.5, 2))
            #              + 0.03*np.exp(-((z-4.5)/2.0**2))) * pow(np.log10(m_mf)-7, 2)

            a_fuv = (0.05 +
                     0.25 * np.exp(-(((redshift-2.0)/1.5)**2)) +
                     0.03 * np.exp(-(((redshift-4.5)/2.0)**2))  ) * (np.log10(m_star)-7)**2
        else:
            a_fuv = 0.025*np.log10(m_star)

        # From Buat et al. (2013, ApJ 807, 141) for 1.2 < z < 4.0
        #a_fuv = 0.89 * np.log10(m_star) - 6.77
        # From Reddy et al. (2016): E(B-V)_gas - E(B-V)_stars = -0.049 + 0.079 / xsi
        # with xsi = 1./(log10(SFR/M_star) +10.)
        # sSFRs = SFR(Halpha=over10Myrs) / Mstar with SFR(Halpha) corrected for dust attenuation
        #self.ebvs_old_factor = -0.049 + 0.079 / (np.log10(sfr10Myrs/m_star) + 10.)

        # From de Barros et al. (2016): E(B-V)_gas - E(B-V)_stars = 0.31 * (log10(sSFR)+9.) - 0.16
        #self.ebvs_old_factor = 0.31 * (np.log10(sfr10Myrs/m_star)+9.) - 0.16
        # From CIGALE: E_BVs_stellar_young ~ A_FUV/k(lambda)*m_star/(m_star_old*ebvs_old_factor+m_star_young)
        # k(lambda) = 10.2257 for lambda = 0.1528 micron

        ebvs['young'] = max(0., (a_fuv/10.2257-0.06)/1.05/self.ebvs_old_factor)
        ebvs['young'] = max(0., 0.13*a_fuv**1.25)
        ebvs['old'] = ebvs['young'] * self.ebvs_old_factor

        # Fλ fluxes (only from continuum) in each filter before attenuation.
        flux_noatt = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

        # Compute attenuation curve
        attenuation_curve = a_vs_ebv(
            wavelength, self.uv_bump_wavelength, self.uv_bump_width,
            self.uv_bump_amplitude, self.powerlaw_slope)

        attenuation_total = 0.
        contribs = [contrib for contrib in sed.contribution_names if
                    'absorption' not in contrib]
        for contrib in contribs:
            age = contrib.split('.')[-1].split('_')[-1]
            luminosity = sed.get_lumin_contribution(contrib)
            attenuated_luminosity = (
                luminosity * 10. ** (ebvs[age] * attenuation_curve / -2.5))
            attenuation_spectrum = attenuated_luminosity - luminosity
            # We integrate the amount of luminosity attenuated (-1 because the
            # spectrum is negative).
            attenuation = -1. * np.trapz(attenuation_spectrum, wavelength)
            attenuation_total += attenuation

            sed.add_module(self.name, self.parameters)
            sed.add_info("attenuation.E_BVs." + contrib, ebvs[age])
            sed.add_info("attenuation." + contrib, attenuation, True)
            sed.add_contribution("attenuation." + contrib, wavelength,
                                 attenuation_spectrum)

        # Total attenuation
        if 'dust.luminosity' in sed.info:
            sed.add_info("dust.luminosity",
                         sed.info["dust.luminosity"]+attenuation_total, True,
                         True)
        else:
            sed.add_info("dust.luminosity", attenuation_total, True)

        # Fλ fluxes (only from continuum) in each filter after attenuation.
        flux_att = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

        # Attenuation in each filter
        for filt in self.filter_list:
            sed.add_info("attenuation." + filt,
                         -2.5 * np.log10(flux_att[filt] / flux_noatt[filt]))

        sed.add_info('attenuation.ebvs_old_factor', self.ebvs_old_factor)
        sed.add_info('attenuation.uv_bump_wavelength', self.uv_bump_wavelength)
        sed.add_info('attenuation.uv_bump_width', self.uv_bump_width)
        sed.add_info('attenuation.uv_bump_amplitude', self.uv_bump_amplitude)
        sed.add_info('attenuation.powerlaw_slope', self.powerlaw_slope)

# SedModule to be returned by get_module
Module = CalzFlare
