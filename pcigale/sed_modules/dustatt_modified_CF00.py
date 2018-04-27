# -*- coding: utf-8 -*-
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt

"""
Modified Charlot & Fall 2000 dust attenuation module
===================================

This module implements an attenuation law combining the birth cloud (BC)
attenuation and the interstellar medium (ISM) attenuation, each one modelled by
a power law. The young star emission is attenuated by the BC and the ISM
attenuations whereas the old star emission is only affected by the ISM. This
simple model was proposed by Charlot & Fall (2000).

Parameters available for analysis
---------------------------------

- attenuation.Av_ISM: Av attenuation in the interstellar medium
- attenuation.mu: Av_ISM / (Av_BV+Av_ISM)
- attenuation.slope_BC: slope of the power law in the birth clouds
- attenuation.slope_ISM: slope of the power law in the ISM
- attenuation.<NAME>: amount of total attenuation in the luminosity
    contribution <NAME>
- attenuation.<FILTER>: total attenuation in the filter
"""

from collections import OrderedDict

import numpy as np

from . import SedModule


def alambda_av(wl, delta):
    """Compute the complete attenuation curve A(λ)/Av

    Attenuation curve of the form (λ / 550 nm) ** δ. The Lyman continuum is not
    attenuated.

    Parameters
    ----------
    wl: array of floats
        The wavelength grid in nm.
    delta: float
        Slope of the power law.

    Returns
    -------
    attenuation: array of floats
        The A(λ)/Av attenuation at each wavelength of the grid.

    """
    attenuation = (wl / 550.) ** delta

    # Lyman continuum not attenuated.
    attenuation[wl <= 91.2] = 0.

    return attenuation


class ModCF00Att(SedModule):
    """Two power laws attenuation module

    Attenuation module combining the birth cloud (BC) attenuation and the
    interstellar medium (ISM) one.

    The attenuation can be computed on the whole spectrum or on a specific
    contribution and is added to the SED as a negative contribution.

    """

    parameter_list = OrderedDict([
        ("Av_ISM", (
            "cigale_list(minvalue=0)",
            "V-band attenuation in the interstellar medium.",
            1.
        )),
        ("mu", (
            "cigale_list(minvalue=.0001, maxvalue=1.)",
            "Av_ISM / (Av_BC+Av_ISM)",
            0.44
        )),
        ("slope_ISM", (
            "cigale_list()",
            "Power law slope of the attenuation in the ISM.",
            -0.7
        )),
        ("slope_BC", (
            "cigale_list()",
            "Power law slope of the attenuation in the birth clouds.",
            -1.3
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
        self.Av_ISM = float(self.parameters['Av_ISM'])
        self.mu = float(self.parameters['mu'])
        self.slope_ISM = float(self.parameters['slope_ISM'])
        self.slope_BC = float(self.parameters['slope_BC'])
        self.filter_list = [item.strip() for item in
                            self.parameters["filters"].split("&")]
        self.Av_BC = self.Av_ISM * (1. - self.mu) / self.mu
        self.curve = {}

    def process(self, sed):
        """Add the dust attenuation to the SED.

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """

        wl = sed.wavelength_grid

        if len(self.curve) == 0:
            self.curve['old'] = 10. ** (-.4 * alambda_av(wl, self.slope_ISM) *
                                        self.Av_ISM)
            # Emission from the young population is attenuated by both
            # components
            self.curve['young'] = 10. ** (-.4 * alambda_av(wl, self.slope_BC) *
                                          self.Av_BC) * self.curve['old']

        # Fλ fluxes in each filter before attenuation.
        flux_noatt = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

        dust_lumin = 0.
        contribs = [contrib for contrib in sed.contribution_names if
                    'absorption' not in contrib]

        for contrib in contribs:
            age = contrib.split('.')[-1].split('_')[-1]
            luminosity = sed.get_lumin_contribution(contrib)

            attenuation_spectrum = luminosity * (self.curve[age] - 1.)
            dust_lumin -= np.trapz(attenuation_spectrum, wl)

            sed.add_module(self.name, self.parameters)
            sed.add_contribution("attenuation." + contrib, wl,
                                 attenuation_spectrum)

        sed.add_info('attenuation.Av_ISM', self.Av_ISM)
        sed.add_info('attenuation.Av_BC', self.Av_BC)
        sed.add_info('attenuation.mu', self.mu)
        sed.add_info('attenuation.slope_BC', self.slope_BC)
        sed.add_info('attenuation.slope_ISM', self.slope_ISM)

        # Total attenuation
        if 'dust.luminosity' in sed.info:
            sed.add_info("dust.luminosity",
                         sed.info["dust.luminosity"] + dust_lumin, True, True)
        else:
            sed.add_info("dust.luminosity", dust_lumin, True)

        # Fλ fluxes (only in continuum) in each filter after attenuation.
        flux_att = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

        # Attenuation in each filter
        for filt in self.filter_list:
            att = -2.5 * np.log10(flux_att[filt] / flux_noatt[filt])
            sed.add_info("attenuation." + filt, max(0., att))


# CreationModule to be returned by get_module
Module = ModCF00Att
