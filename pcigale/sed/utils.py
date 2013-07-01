# -*- coding: utf-8 -*-
"""
Copyright (C) 2012, 2013 Centre de données Astrophysiques de Marseille
Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt

@author: Yannick Roehlly <yannick.roehlly@oamp.fr>
@author: Médéric Boquien <mederic.boquien@oamp.fr>

"""


import numpy as np
from scipy import integrate
from scipy.constants import c, pi, parsec


def lambda_to_nu(wavelength):
    """Convert wavelength (nm) to frequency (Hz)

    Parameters
    ----------
    wavelength : float or array of floats
        The wavelength(s) in nm.

    Returns
    -------
    nu : float or array of floats
        The frequency(ies) in Hz.

    """
    return c / (wavelength * 1.e-9)


def nu_to_lambda(frequency):
    """Convert frequency (Hz) to wavelength (nm)

    Parameters
    ----------
    frequency : float or numpy.array of floats
        The frequency(ies) in Hz.

    Returns
    -------
    wavelength : float or numpy.array of floats
        The wavelength(s) in nm.

    """
    return 1.e-9 * c / frequency


def best_grid(wavelengths1, wavelengths2):
    """
    Return the best wavelength grid to regrid to arrays

    Considering the two wavelength grids passed in parameters, this function
    compute the best new grid that will be used to regrid the two spectra
    before combining them.

    Parameters
    ----------
    wavelengths1, wavelengths2 : array of floats
        The wavelength grids to be 'regrided'.

    Returns
    -------
    new_grid : array of floats
        Array containing all the wavelengths found in the input arrays.

    """
    new_grid = np.hstack((wavelengths1, wavelengths2))
    new_grid.sort()
    new_grid = np.unique(new_grid)

    return new_grid


def luminosity_distance(z, h0=71., omega_m=0.27, omega_l=0.73):
    """
    Computes luminosity distance at redshift z in Mpc for given Λ cosmology
    (H_0 in (km/s)/Mpc, Ω_M, and Ω_Λ) Ref.: Hogg (1999) astro-ph/9905116

    Parameters
    ----------
    z : float
        Redshift
    h0 : float
        Hubble's constant
    omega_m : float
        Omega matter.
    omega_l : float
        Omega vacuum

    Returns
    -------
    luminosity_distance : float
        The luminosity distance in Mpc.

    """

    omega_k = 1. - omega_m - omega_l

    if z > 0.:
        dist, edist = integrate.quad(
            lambda x: (omega_m * (1. + x) ** 3
                       + omega_k * (1 + x) ** 2 + omega_l) ** (-.5),
            0.,
            z,
            epsrel=1e-3)
    else:
        # Bad idea as there is something *wrong* going on
        print('LumDist: z <= 0 -> Assume z = 0!')
        z = 0.
        dist = 0.

    if omega_k > 0.:
        dist = np.sinh(dist * np.sqrt(omega_k)) / np.sqrt(omega_k)
    elif omega_k < 0.:
        dist = np.sin(dist * np.sqrt(-omega_k)) / np.sqrt(-omega_k)

    return c / (h0 * 1.e3) * (1. + z) * dist


def luminosity_to_flux(luminosity, redshift=0):
    """
    Convert a luminosity (or luminosity density) to a flux (or flux density).

    F = L / (4πDl2)

    Parameters
    ----------
    luminosity : float or array of floats
        Luminosity (typically in W) or luminosity density (W/nm or W/Hz).
    redshift :
        Redshift. If redshift is 0 (the default) the flux at a luminosity
        distance of 10 pc is returned.

    Returns
    -------
    flux : float or array of floats
        The flux (typically in W/m²) of flux density (W/m²/nm or W/m²/Hz).

    """
    if redshift == 0:
        dist = 10 * parsec
    else:
        dist = luminosity_distance(redshift) * 1.e6 * parsec

    return luminosity / (4 * pi * np.square(dist))


def lambda_flambda_to_fnu(wavelength, flambda):
    """
    Convert a Fλ vs λ spectrum to Fν vs λ

    Parameters
    ----------
    wavelength : list-like of floats
        The wavelengths in nm.
    flambda : list-like of floats
        Fλ flux density in W/m²/nm (or Lλ luminosity density in W/nm).

    Returns
    -------
    fnu : array of floats
        The Fν flux density in mJy (or the Lν luminosity density in
        1.e-29 W/Hz).

    """
    wavelength = np.array(wavelength, dtype=float)
    flambda = np.array(flambda, dtype=float)

    # Factor 1e+29 is to switch from W/m²/Hz to mJy
    # Factor 1e-9 is to switch from nm to m (only one because the other nm
    # wavelength goes with the Fλ in W/m²/nm).
    fnu = 1e+29 * 1e-9 * flambda * wavelength * wavelength / c

    return fnu


def lambda_fnu_to_flambda(wavelength, fnu):
    """
    Convert a Fν vs λ spectrum to Fλ vs λ

    Parameters
    ----------
    wavelength : list-like of floats
        The wavelengths in nm.
    fnu : list-like of floats
        The Fν flux density in mJy (of the  Lν luminosity density in
        1.e-29 W/Hz).

    Returns
    -------
    flambda : array of floats
        Fλ flux density in W/m²/nm (or Lλ luminosity density in W/nm).

    """
    wavelength = np.array(wavelength, dtype=float)
    fnu = np.array(fnu, dtype=float)

    # Factor 1e-29 is to switch from Jy to W/m²/Hz
    # Factor 1e+9 is to switch from m to nm
    flambda = 1e-29 * 1e+9 * fnu / (wavelength * wavelength) * c

    return flambda


def redshift_spectrum(wavelength, flux, redshift, is_fnu=False):
    """Redshit a spectrum

    Parameters
    ----------
    wavelength : array like of floats
        The wavelength in nm.
    flux : array like of floats
        The flux or luminosity density.
    redshift : float
        The redshift.
    is_fnu : boolean
        If false (default) the flux is a Fλ density in W/m²/nm (or a Lλ
        luminosity density in W/nm). If true, the flux is a Fν density in mJy
        (or a Lν luminosity density in 1.e-29 W/Hz).

    Results
    -------
    wavelength, flux : tuple of numpy arrays of floats
        The redshifted spectrum with the same kind of flux (or luminosity)
        density as the input.

    """
    wavelength = np.array(wavelength, dtype=float)
    flux = np.array(flux, dtype=float)
    redshift = float(redshift)

    if redshift < 0:
        redshift_factor = 1. / (1. - redshift)
    else:
        redshift_factor = 1. + redshift

    if is_fnu:
        # Switch to Fλ
        flux = lambda_fnu_to_flambda(wavelength, flux)

    wavelength *= redshift_factor
    flux /= redshift_factor

    if is_fnu:
        # Switch back to Fλ
        flux = lambda_flambda_to_fnu(wavelength, flux)

    return wavelength, flux
