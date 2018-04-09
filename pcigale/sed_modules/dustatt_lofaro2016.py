# -*- coding: utf-8 -*-
# Copyright (C) 2016 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly, Barabara Lo Faro, Véronique Buat

"""
Lo Faro+ 2016 attenuation module
=================================

This module implements an attenuation law following the prescription of
Lo Faro et al (2016). This is a piecewise-defined function composed of a power
law with a blue slope for wavelengths before 5500 Å, and a power law with
a different red slope after.

Here are some values of blue slope, red slope that mimics common attenuation
laws:

- Calzetti: -0.7, -1.4
- SMC: -1.2, -1.7
- Charlot and Fall: -0.7, -0.7 (assuming the same attenuation for old and young
  stars)
- Lo Faro 2016: -0.48, -0.48 (assuming the same attanuation for old and young
  stars)

Parameters available for analysis
---------------------------------

- attenuation.Av: Av attenuation
- attenuation.blue_slope: slope of the power law used before 5500 Å
- attenuation.red_slope: slope of the power law used after 5500 Å
- attenuation.<NAME>: amount of total attenuation in the luminosity
    contribution <NAME>
- attenuation.<FILTER>: total attenuation in the filter
"""

from collections import OrderedDict

import numpy as np

from . import SedModule


def alambda_av(wavelengths, delta_blue, delta_red):
    """Compute the complete attenuation curve A(λ)/Av

    The attenuation curve is a piecewise-defined function composed of two power
    laws (λ / λv) ** δ with two different slopes before and after 550 nm.

    The Lyman continuum is not attenuated.

    Parameters
    ----------
    wavelengths: array of floats
        The wavelength grid (in nm) to compute the attenuation curve on.
    delta_blue: float
        Slope of the power law before 550 nm.
    delta_red: float
        Slope of the power law after 550 nm.

    Returns
    -------
    attenuation: array of floats
        The A(λ)/Av attenuation at each wavelength of the grid.

    """
    wave = np.array(wavelengths)
    attenuation = np.zeros_like(wave)

    # Between Lyman break and 550 nm
    mask = (wave > 91.2) & (wave <= 550)
    attenuation[mask] = (wave[mask] / 550) ** delta_blue

    # After 550 nm
    mask = wave > 550
    attenuation[mask] = (wave[mask] / 550) ** delta_red

    return attenuation


class Lofaro2016Att(SedModule):
    """Lo Faro et al (2016) attenuation module

    Attenuation module implementing the prescription of Lo Faro et al (2016).
    The same attenuation law is applied to young and old star populations. This
    law is a piecewise-defined function compose of two power laws, one before
    5500 Å, and the other after.

    The attenuation is added to the SED as a negative contribution.

    """

    parameter_list = OrderedDict([
        ("Av", (
            "cigale_list(minvalue=0)",
            "V-band attenuations.",
            1.
        )),
        ("slopes", (
            "cigale_list(dtype=str)",
            "Slopes for the power low separated by a &: first the blue slope "
            "(used before 5500 Å), then the red slope. Reference values are: "
            "-0.7 & -1.4 for Calzetti, -1.2 & -1.7 for SMC, and assuming the "
            "same attenuation for young and old stars -0.7 & -0.7 for Charlot "
            "and Fall, and -0.48 & -0.48 for Lo Faro (2016). Several slope "
            "tuples may be given separated with comma.",
            "-0.7 & -1.4"
        )),
        ("filters", (
            "string()",
            "Filters for which the attenuation will be computed and added to "
            "the SED information dictionary. You can give several filter "
            "names separated by a & (don't use commas).",
            "V_B90 & FUV"
        ))
    ])

    def _init_code(self):
        self.av = float(self.parameters["Av"])
        self.blue_slope, self.red_slope = [
            float(item) for item in self.parameters["slopes"].split("&")
        ]
        self.filter_list = [item.strip() for item in
                            self.parameters["filters"].split("&")]

    def process(self, sed):
        """Add the dust attenuation to the SED.

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """

        wavelength = sed.wavelength_grid

        # Fλ fluxes in each filter before attenuation.
        flux_noatt = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

        attenuation_total = 0.
        contribs = [contrib for contrib in sed.contribution_names if
                    'absorption' not in contrib]

        for contrib in contribs:

            luminosity = sed.get_lumin_contribution(contrib)

            extinction_factor = 10 ** (
                self.av * alambda_av(wavelength, self.blue_slope,
                                     self.red_slope) / -2.5)

            attenuated_luminosity = luminosity * extinction_factor
            attenuation_spectrum = attenuated_luminosity - luminosity
            # We integrate the amount of luminosity attenuated (-1 because the
            # spectrum is negative).
            attenuation = -1 * np.trapz(attenuation_spectrum, wavelength)
            attenuation_total += attenuation

            sed.add_module(self.name, self.parameters)
            sed.add_info("attenuation." + contrib, attenuation, True)
            sed.add_contribution("attenuation." + contrib, wavelength,
                                 attenuation_spectrum)

        sed.add_info('attenuation.Av', self.av)
        sed.add_info('attenuation.blue_slope', self.blue_slope)
        sed.add_info('attenuation.red_slope', self.red_slope)

        # Total attenuation
        if 'dust.luminosity' in sed.info:
            sed.add_info("dust.luminosity",
                         sed.info["dust.luminosity"]+attenuation_total, True,
                         True)
        else:
            sed.add_info("dust.luminosity", attenuation_total, True)

        # Fλ fluxes (only in continuum) in each filter after attenuation.
        flux_att = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

        # Attenuation in each filter
        for filt in self.filter_list:
            sed.add_info("attenuation." + filt,
                         -2.5 * np.log10(flux_att[filt] / flux_noatt[filt]))

# CreationModule to be returned by get_module
Module = Lofaro2016Att
