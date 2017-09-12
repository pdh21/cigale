# -*- coding: utf-8 -*-
# Copyright (C) 2014 Institute of Astronomy
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Médéric Boquien

import numpy as np


class SB99(object):
    """Single Stellar Population as defined in Starburst 99

    This class holds the data associated with a single stellar population
    (SSP) as defined in Starburst 99.

    """

    def __init__(self, imf, metallicity, rotation, time_grid, wavelength_grid,
                 info_table, spec_table):
        """Create a new single stellar population as defined in Starburst 99.

        Parameters
        ----------
        imf: string
            Initial mass function (IMF): either 'salp' for Salpeter (1955) or
            'krou' for Kroupa (2001).
        metallicity: float
            The metallicity. Possible values are 0.002, 0.014.
        time_grid: array of floats
            The time [Myr] grid used in the colors_table and the lumin_table.
        wavelength_grid: array of floats
            The wavelength [nm] grid used in the lumin_table.
        color_table: 2 axis array of floats
            The wavelength [nm] grid used in the lumin_table.
            - n_ly: the number of Lyman continuum photons in s¯¹
            - m_star: the stellar mass in Msun
        lumin_table: 2 axis array of floats
            Luminosity density in W/nm. The first axis is the wavelength and
            the second the time (index bases on the wavelength and time grids).
        """

        if imf in ['salp', 'krou']:
            self.imf = imf
        else:
            raise ValueError('IMF must be either sal for Salpeter or '
                             'krou for Kroupa.')
        self.metallicity = metallicity
        self.rotation = rotation
        self.time_grid = time_grid
        self.wavelength_grid = wavelength_grid
        self.info_table = info_table
        self.spec_table = spec_table

    def convolve(self, sfh, separation_age):
        """Convolve the SSP with a Star Formation History

        Given a SFH (an time grid and the corresponding star formation rate
        SFR), this method convolves the color table and the SSP luminosity
        spectrum along the whole SFR.

        The time grid of the SFH is expected to be ordered and must not run
        beyong 14 Gyr (the maximum time for Starburst 99 SSP).

        Parameters
        ----------
        sfh: array of floats
            Star Formation History in Msun/yr.
        separation_age: float
            Age separating the young from the old stellar populations in Myr.

        Returns
        -------
        wavelength: array of floats
            Wavelength grid [nm] for the spectrum
        luminosity: array of floats
            Luminosity density [W/nm] at each wavelength.
        sb99_info: dictionary
             Dictionary containing various information:
             - n_ly: the number of Lyman continuum photons in s¯¹
             - m_star: the stellar mass in Msun

        """
        # As both the SFH and the SSP (limited to the age of the SFH) data now
        # share the same time grid, the convolution is just a matter of
        # reverting one and computing the sum of the one to one product; this
        # is done using the dot product.
        info_table = self.info_table[:, :sfh.size]
        spec_table = self.spec_table[:, :sfh.size]

        # The convolution is just a matter of reverting the SFH and computing
        # the sum of the data from the SSP one to one product. This is done
        # using the dot product. The 1e6 factor is because the SFH is in solar
        # mass per year.
        info_young = 1e6 * np.dot(info_table[:, :separation_age],
                                  sfh[-separation_age:][::-1])
        spec_young = 1e6 * np.dot(spec_table[:, :separation_age],
                                  sfh[-separation_age:][::-1])

        info_old = 1e6 * np.dot(info_table[:, separation_age:],
                                sfh[:-separation_age][::-1])
        spec_old = 1e6 * np.dot(spec_table[:, separation_age:],
                                sfh[:-separation_age][::-1])

        info_all = info_young + info_old

        info_all = info_young + info_old

        info_young = dict(zip(["n_ly", "m_star"], info_young))
        info_old = dict(zip(["n_ly", "m_star"], info_old))
        info_all = dict(zip(["n_ly", "m_star"], info_all))

        info_all['age_mass'] = np.average(self.time_grid[:sfh.size],
                                          weights=info_table[0, :] * sfh[::-1])


        return spec_young, spec_old, info_young, info_old, info_all
