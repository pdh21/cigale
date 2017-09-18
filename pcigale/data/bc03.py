# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly

import numpy as np


class BC03(object):
    """Single Stellar Population as defined in Bruzual and Charlot (2003)

    This class holds the data associated with a single stellar population
    (SSP) as defined in Bruzual and Charlot (2003). Compared to the pristine
    Bruzual and Charlot SSP:
        * The time grid ranges from 1 Myr to 14 Gyr with 1 Myr steps.
        * The SSP are all interpolated on this new grid.
        * The wavelength grid is refined beyond 10 μm to avoid artefacts.
        * The wavelength is given in nm rather than Å.
        * The spectra are given in W/nm rather than Lsun.

    """

    def __init__(self, imf, metallicity, time_grid, wavelength_grid,
                 info_table, spec_table):
        """Create a new single stellar population as defined in Bruzual and
        Charlot (2003).

        Parameters
        ----------
        imf: string
            Initial mass function (IMF): either 'salp' for Salpeter (1955) or
            'chab' for Chabrier (2003).
        metallicity: float
            The metallicity. Possible values are 0.0001, 0.0004, 0.004, 0.008,
            0.02, and 0.05.
        time_grid: array of floats
            The time grid in Myr used in the info_table and the spec_table.
        wavelength_grid: array of floats
            The wavelength grid in nm used in spec_table.
        info_table: 2 axis array of floats
            Array containing information from some of the *.?color tables from
            Bruzual and Charlot (2003) at each time of the time_grid.
                * info_table[0]: Total mass in stars in solar mass
                * info_table[1]: Mass returned to the ISM by evolved stars in
                    solar mass
                * info_table[2]: rate of H-ionizing photons (s-1)
        spec_table: 2D array of floats
            Spectrum of the SSP in W/nm (first axis) every 1 Myr (second axis).

        """

        if imf in ['salp', 'chab']:
            self.imf = imf
        else:
            raise ValueError('IMF must be either sal for Salpeter or '
                             'cha for Chabrier.')
        self.metallicity = metallicity
        self.time_grid = time_grid
        self.wavelength_grid = wavelength_grid
        self.info_table = info_table
        self.spec_table = spec_table

    def convolve(self, sfh, separation_age):
        """Convolve the SSP with a Star Formation History

        Given an SFH, this method convolves the info table and the SSP
        luminosity spectrum.

        Parameters
        ----------
        sfh: array of floats
            Star Formation History in Msun/yr.
        separation_age: float
            Age separating the young from the old stellar populations in Myr.

        Returns
        -------
        spec_young: array of floats
            Spectrum in W/nm of the young stellar populations.
        spec_old: array of floats
            Same as spec_young but for the old stellar populations.
        info_young: dictionary
            Dictionary containing various information from the *.?color tables
            for the young stellar populations:
            * "m_star": Total mass in stars in Msun
            * "m_gas": Mass returned to the ISM by evolved stars in Msun
            * "n_ly": rate of H-ionizing photons (s-1)
        info_old : dictionary
            Same as info_young but for the old stellar populations.
        info_all: dictionary
            Same as info_young but for the entire stellar population. Also
            contains "age_mass", the stellar mass-weighted age

        """
        # We cut the SSP to the maximum age considered to simplify the
        # computation. We take only the first three elements from the
        # info_table as the others do not make sense when convolved with the
        # SFH (break strength).
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

        info_young = dict(zip(["m_star", "m_gas", "n_ly"], info_young))
        info_old = dict(zip(["m_star", "m_gas", "n_ly"], info_old))
        info_all = dict(zip(["m_star", "m_gas", "n_ly"], info_all))

        info_all['age_mass'] = np.average(self.time_grid[:sfh.size],
                                          weights=info_table[0, :] * sfh[::-1])

        return spec_young, spec_old, info_young, info_old, info_all
