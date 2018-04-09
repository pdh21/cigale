# -*- coding: utf-8 -*-
# Copyright (C) 2015 Laboratoire d'Astrophysique de Marseille
# Copyright (C) 2016 Universidad de Antofagasta
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Médéric Boquien & Denis Burgarella

"""
Module that estimates other parameters, e.g., UV slope, Lick indices, etc.
==========================================================================

This module estimates additional parameters of interest close to the
observation, e.g., the ultraviolet slope (beta), the rest-frame, any type of
indices (Lick), etc.

This module has to be called right before redshifting as such it does not take
into account the IGM absorption so it should not be used to compute fluxes in
the observed frame. For that use the param_postz module.

"""

from collections import OrderedDict
from itertools import chain

import numpy as np
from scipy.constants import c, parsec

from . import SedModule
from ..sed.utils import flux_trapz


class RestframeParam(SedModule):
    """Compute miscellaneous parameters on the full SED. This is a separate
    module as computing these quantitites in other SED modules does not always
    make much sense. This module is to be called right before the redshifting
    module. This means it does not include IGM absorbtion

    """

    parameter_list = OrderedDict([
        ("beta_calz94", (
            "boolean()",
            "UV slope measured in the same way as in Calzetti et al. (1994).",
            False
        )),
        ("D4000", (
            "boolean()",
            "D4000 break using the Balogh et al. (1999) definition.",
            False
        )),
        ("IRX", (
            "boolean()",
            "IRX computed from the GALEX FUV filter and the dust luminosity.",
            False
        )),
        ("EW_lines", (
            "string()",
            "Central wavelength of the emission lines for which to compute "
            "the equivalent width. The half-bandwidth must be indicated "
            "after the '/' sign. For instance 656.3/1.0 means oth the nebular "
            "line and the continuum are integrated over 655.3-657.3 nm.",
            "500.7/1.0 & 656.3/1.0"
        )),
        ("luminosity_filters", (
            "string()",
            "Filters for which the rest-frame luminosity will be computed. "
            "You can give several filter names separated by a & (don't use "
            "commas).",
            "FUV & V_B90"
        )),
        ("colours_filters", (
            "string()",
            "Rest-frame colours to be computed. You can give several colours "
            "separated by a & (don't use commas).",
            "FUV-NUV & NUV-r_prime"
        )),
        ("Flux_emission_lines", (
            "string()",
            "Name of the emission lines for which to compute the flux "
            "separated by a & (don't use commas).",
            "Halpha & Hbeta"
        ))
    ])

    def calz94(self, sed):
        wl = sed.wavelength_grid
        lumin = sed.luminosity

        # Attenuated (observed) UV slopes beta as defined in Calzetti et al.
        # (1994, ApJ 429, 582, Tab. 2) that excludes the 217.5 nm bump
        # wavelength range and other spectral features
        if 'nebular.lines_width' in sed.info:
            key = (wl.size, sed.info['nebular.lines_width'])
        else:
            key = (wl.size, )

        if key in self.w_calz94:
            w_calz94 = self.w_calz94[key]
        else:
            calz_wl = [(126.8, 128.4), (130.9, 131.6), (134.2, 137.1),
                       (140.7, 151.5), (156.2, 158.3), (167.7, 174.0),
                       (176.0, 183.3), (186.6, 189.0), (193.0, 195.0),
                       (240.0, 258.0)]
            w_calz94 = np.where(np.any([(wl >= wlseg[0]) & (wl <= wlseg[1])
                                        for wlseg in calz_wl], axis=0))

            self.w_calz94[key] = w_calz94

        # We compute the regression directly from the covariance matrix as the
        # numpy/scipy regression routines are quite slow.
        ssxm, ssxym, _, _ = np.cov(np.log10(wl[w_calz94]),
                                   np.log10(lumin[w_calz94]),
                                   bias=1).flat

        return ssxym / ssxm

    def D4000(self, sed):
        wl = sed.wavelength_grid
        fnu = sed.fnu

        # Strength of the D_4000 break using Balogh et al. (1999, ApJ 527, 54),
        # i.e., ratio of the flux in the red continuum to that in the blue
        # continuum: Blue continuum: 385.0-395.0 nm & red continuum:
        # 400.0-410.0 nm.
        if 'nebular.lines_width' in sed.info:
            key = (wl.size, sed.info['nebular.lines_width'])
        else:
            key = (wl.size, )

        if key in self.w_D4000blue:
            w_D4000blue = self.w_D4000blue[key]
            w_D4000red = self.w_D4000red[key]
        else:
            w_D4000blue = np.where((wl >= 385.0) & (wl <= 395.0))
            w_D4000red = np.where((wl >= 400.0) & (wl <= 410.0))
            self.w_D4000blue[key] = w_D4000blue
            self.w_D4000red[key] = w_D4000red

        return (np.trapz(fnu[w_D4000red], x=wl[w_D4000red]) /
                np.trapz(fnu[w_D4000blue], x=wl[w_D4000blue]))

    def EW(self, sed):
        wl = sed.wavelength_grid

        key = (wl.size, sed.info['nebular.lines_width'])
        if key in self.w_lines:
            w_lines = self.w_lines[key]
        else:
            w_lines = {line: np.where((wl >= line[0]-line[1]) &
                                      (wl <= line[0]+line[1]))
                       for line in self.lines}
            self.w_lines[key] = w_lines

        lumin_line = np.sum([sed.get_lumin_contribution(name)
                             for name in sed.contribution_names
                             if 'nebular.lines' in name], axis=0)
        lumin_cont = sed.luminosity - lumin_line

        EW = {}
        for line in self.lines:
            w_line = w_lines[line]
            wl_line = wl[w_line]
            key = (wl_line.size, sed.info['nebular.lines_width'], line[0], 0.)
            EW[line] = (flux_trapz(lumin_line[w_line], wl_line, key) /
                        flux_trapz(lumin_cont[w_line], wl_line, key) *
                        (wl_line[-1]-wl_line[0]))

        return EW

    def emission_lines_library(self):
        """ Creates a dictionary containing all available emission lines.
        Central wavelength express in Angstroms.
        Values taken taken from database_builder/nebular/line_wavelengths.dat
        """

        lib = dict(Ly_alpha=1216,
                   CII_1335=1335,
                   SiIV=1397,
                   CIV=1549,
                   HeII=1640,
                   OIII_1665=1665,
                   CIII=1909,
                   CII_2326=2326,
                   MgII=2798,
                   OII=3727,
                   Htheta=3798,
                   NeIII=3869,
                   Heta=3835,
                   HeI=3889,
                   Hepsilon=3970,
                   SII=4070,
                   Hdelta=4102,
                   Hgamma=4340,
                   Hbeta=4861,
                   OIII_4959=4959,
                   OIII_5007=5007,
                   OI_6300=6300,
                   NII_6548=6548,
                   Halpha=6563,
                   NII_6584=6584,
                   SII_6716=6716,
                   SII_6731=6731)

        return lib

    def EL_flux(self, sed):
        """ Compute the flux of emission lines given in the properties.
        This is done on the rest-frame SED.
        The Flux is integrated when Lumin_tot > Lumin_cont around wvl
        of interest. Underlying stellar absoprtion is theoretically
        taken into account, meaning that the flux is corrected for
        underlying stellar absorption.
        Computed with CIGALE resolution, not at instrument resolution
        """

        # Wavelength in nm
        wl = sed.wavelength_grid
        # Luminosites in W / nm
        lumin_lines = np.sum([sed.get_lumin_contribution(name)
                             for name in sed.contribution_names
                             if 'nebular.lines' in name], axis=0)

        lumin_tot = sed.luminosity
        lumin_cont = sed.luminosity - lumin_lines

        # Get the names of lines
        EL_names = [item.strip() for item in
                    self.parameters['Flux_emission_lines'].split("&")
                    if item.strip() != '']

        # Create dictionary to store the fluxes
        EL_fluxes = {}

        # Check whether there are emission lines to compute
        if EL_names != []:
            for EL_name in EL_names:
                # Check whether lines should be added
                Flux_EL = 0

                for i, EL_ in enumerate(EL_name.split("+")):
                    EL_ = EL_.strip()

                    # Get central wavelength in nm
                    wvl_c = self.EL_lib[EL_]
                    wvl_c *= 0.1

                    # Get wavelength interval where:
                    # Lum_tot > Lum_tot - Lum_lines

                    # Nebular luminosity is never 0.
                    # so we need to multiply by a small factor to use:
                    # Lum_cont = Lum_tot - Lum_lines
                    # Lum_tot > Lum_cont * (1+factor)
                    factor = 1e-3

                    # get array index for central wavelength
                    line_idx = np.argmin((wl-wvl_c)**2)
                    # go backward to get wvl_min
                    idx = line_idx
                    wvl_min = np.nan
                    while lumin_tot[idx] > lumin_cont[idx]*(1+factor):
                        idx = idx-1
                    wvl_min = wl[idx+1]
                    # get wvl_max
                    idx = line_idx
                    wvl_max = np.nan
                    while lumin_tot[idx] > lumin_cont[idx]*(1+factor):
                        idx += 1
                    wvl_max = wl[idx-1]

                    # Sometimes code stops because no wvl_min is found
                    if np.isfinite(wvl_min) and np.isfinite(wvl_max):
                        mask_EL = (wl >= wvl_min) & (wl <= wvl_max)
                        Flux_new = np.trapz(lumin_lines[mask_EL], wl[mask_EL])
                        if Flux_new == Flux_EL:
                            print ('WARNING: %s flux == %s flux '
                                   % (EL_name.split("+")[i].strip(),
                                      EL_name.split("+")[i-1].strip()))
                            print ('\t %s Flux not added in %s total Flux'
                                   % (EL_name.split("+")[i].strip(),
                                      EL_name))
                        else:
                            Flux_EL += Flux_new
                    else:
                        Flux_EL += -np.inf

                # Total integrated fluxes are expressed in W
                # Will be corrected for Luminosity distance in pdf_analysis
                EL_fluxes['EL_flux_%s' % EL_name] = Flux_EL

        return EL_fluxes

    def _init_code(self):
        # Index of the wavelengths of interest. We use a dictionary with the
        # size of the wavelength array as a key to take into account that
        # different models may have a different sampling, for instance when
        # an AGN may be present or not.
        self.w_calz94 = {}
        self.w_D4000blue = {}
        self.w_D4000red = {}
        self.w_lines = {}

        # Load library of emission lines to associate a name to its wavelength.
        self.EL_lib = self.emission_lines_library()

        # Extract the list of lines to compute the equivalent width
        self.lines = [item.strip() for item in
                      self.parameters["EW_lines"].split("&")
                      if item.strip() != '']
        self.lines = [(float(line.split('/')[0]),
                      float(line.split('/')[1]))
                      for line in self.lines]

        # Extract the list of filters to compute the luminosities
        self.lumin_filters = [item.strip() for item in
                              self.parameters["luminosity_filters"].split("&")
                              if item.strip() != '']

        # Extract the list of rest-frame colours
        self.colours = [item.strip().split("-") for item in
                        self.parameters["colours_filters"].split("&")
                        if item.strip() != '']

        # Extract the list of rest-frame colour filters
        self.colour_filters = list(set(chain(*self.colours)))

        # Compute the list of unique filters to compute both the luminosities
        # and the colours
        self.filters = list(set(self.lumin_filters + self.colour_filters +
                                (["FUV"] if self.parameters["IRX"] is True
                                 else [])))

        # Conversion factor to go from Fnu in mJy to Lnu in W
        self.to_lumin = 1e-29 * 4. * np.pi * (10. * parsec)**2

    def process(self, sed):
        """Computes the parameters for each model.

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """
        fluxes = {filt: sed.compute_fnu(filt) for filt in self.filters}

        if self.parameters['beta_calz94']:
            sed.add_info("param.beta_calz94", self.calz94(sed))
        if self.parameters['D4000']:
            sed.add_info("param.D_4000", self.D4000(sed))
        if self.parameters['IRX']:
            sed.add_info("param.IRX", np.log10(sed.info['dust.luminosity'] /
                         (fluxes['FUV'] * self.to_lumin * c / 154e-9)))

        if 'nebular.lines_young' in sed.contribution_names:
            for line, EW in self.EW(sed).items():
                sed.add_info("param.EW({}/{})".format(*line), EW)

        for filt in self.lumin_filters:
            sed.add_info("param.restframe_Lnu({})".format(filt),
                         fluxes[filt] * self.to_lumin,
                         True)
        for filt1, filt2 in self.colours:
            sed.add_info("param.restframe_{}-{}".format(filt1, filt2),
                         2.5 * np.log10(fluxes[filt2]/fluxes[filt1]))

        for line, flux in self.EL_flux(sed).items():
            sed.add_info("{}".format(line), flux, True)

# SedModule to be returned by get_module
Module = RestframeParam
