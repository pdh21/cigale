# -*- coding: utf-8 -*-
# Copyright (C) 2017 Universidad de Antofagasta
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Médéric Boquien

from astropy.table import Column
import numpy as np

from ..utils import read_table

class ObservationsManager(object):
    """Class to abstract the handling of the observations and provide a
    consistent interface for the rest of cigale to deal with observations.

    An ObservationsManager is in charge of reading the input file to memory,
    check the consistency of the data, replace invalid values with NaN, etc.

    """
    def __new__(cls, config, **kwargs):
        if config['data_file']:
            return ObservationsManagerPassbands(config, **kwargs)
        else:
            return ObservationsManagerVirtual(config, **kwargs)


class ObservationsManagerPassbands(object):
    """Class to generate a manager for data files providing fluxes in
    passbands.

    A class instance can be used as a sequence. In that case a row is returned
    at each iteration.
    """

    def __init__(self, config, defaulterror=0.1, modelerror=0.1,
                 threshold=-9990.):

        self.table = read_table(config['data_file'])
        self.bands = [band for band in config['bands'] if not
                      band.endswith('_err')]
        self.errors = [band for band in config['bands'] if
                       band.endswith('_err')]

        # Sanitise the input
        self._check_filters()
        self._check_errors(defaulterror)
        self._check_invalid(config['analysis_params']['lim_flag'],
                            threshold)
        self._add_model_error(modelerror)

    def __len__(self):
        return len(self.table)

    def __iter__(self):
        self.idx = 0
        self.max = len(self.table)

        return self

    def __next__(self):
        if self.idx < self.max:
            obs = self.table[self.idx]
            self.idx += 1
            return obs
        raise StopIteration

    def _check_filters(self):
        """Check whether the list of filters makes sense.

        Two situations are checked:
        * If a filter to be included in the fit is missing from the data file,
        an exception is raised.
        * If a filter is given in the input file but is not to be included in
        the fit, a warning is displayed

        """
        for band in self.bands + self.errors:
            if band not in self.table.colnames:
                raise Exception("{} to be taken in the fit but not present "
                                "in the observation table.".format(band))

        for band in self.table.colnames:
            if (band != 'id' and band != 'redshift' and
                    band not in self.bands + self.errors):
                self.table.remove_column(band)
                print("Warning: {} in the input file but not to be taken into "
                      "account in the fit.")

    def _check_errors(self, defaulterror=0.1):
        """Check whether the error columns are present. If not, add them.

        This happens in two cases. Either when the error column is not in the
        list of bands to be analysed or when the error column is not present
        in the data file. Note that an error cannot be included explicitly if
        it is not present in the input file. Such a case would be ambiguous
        and will have been caught by self._check_filters().

        We take the error as defaulterror × flux, so by default 10% of the
        flux. The absolute value of the flux is taken in case it is negative.

        Parameters
        ----------
        defaulterror: float
            Fraction of the flux to take as the error when the latter is not
            given in input. By default: 10%.

        """
        if defaulterror < 0.:
            raise ValueError("The relative default error must be positive.")

        for band in self.bands:
            banderr = band + '_err'
            if banderr not in self.errors or banderr not in self.table.colnames:
                colerr = Column(data=np.fabs(self.table[band] * defaulterror),
                                name=banderr)
                self.table.add_column(colerr,
                                      index=self.table.colnames.index(band)+1)
                print("Warning: {}% of {} taken as errors.".format(defaulterror *
                                                                   100.,
                                                                   band))

    def _check_invalid(self, upperlimits=False, threshold=-9990.):
        """Check whether invalid data are correctly marked as such.

        This happens in two cases:
        * Data are marked as upper limits (negative error) but the upper
        limit flag is set to False.
        * Data or errors are lower than -9990.

        We mark invalid data as np.nan. In case the entire column is made of
        invalid data, we remove them from the table

        Parameters
        ----------
        threshold: float
            Threshold under which the data are consisdered invalid.

        """
        allinvalid = []

        for band in self.bands:
            banderr = band + '_err'
            w = np.where((self.table[band] < threshold) |
                         (self.table[banderr] < threshold))
            self.table[band][w] = np.nan
            if upperlimits is False:
                w = np.where(self.table[banderr] < 0.)
                self.table[band][w] = np.nan
            if np.all(~np.isfinite(self.table[band])):
                allinvalid.append(band)

        for band in allinvalid:
            self.bands.remove(band)
            self.errors.remove(band + '_err')
            self.table.remove_columns([band, band + '_err'])
            print("Warning: {} removed as no valid data was found.".format(allinvalid))

    def _add_model_error(self, modelerror=0.1):
        """Add in quadrature the error of the model to the input error.

        Parameters
        ----------
        modelerror: float
            Relative error of the models relative to the flux. By default 10%.

        """
        if modelerror < 0.:
            raise ValueError("The relative model error must be positive.")

        for band in self.bands:
            banderr = band + '_err'
            w = np.where(self.table[banderr] >= 0.)
            self.table[banderr][w] = np.sqrt(self.table[banderr][w]**2. +
                                             (self.table[band][w]*modelerror)**2.)

    def generate_mock(self, fits):
        """Replaces the actual observations with a mock catalogue. It is
        computed from the best fit fluxes of a previous run. For each object
        and each band, we randomly draw a new flux from a Gaussian distribution
        centered on the best fit flux and with a standard deviation identical
        to the observed one.

        Parameters
        ----------
        fits: ResultsManager
            Best fit fluxes of a previous run.

        """
        for idx, band in enumerate(self.bands):
            banderr = band + '_err'
            self.table[band] = np.random.normal(fits.best.fluxes[:, idx],
                                                np.fabs(self.table[banderr]))

    def save(self, filename):
        """Saves the observations as seen internally by the code so it is easy
        to see what fluxes are actually used in the fit. Files are saved in
        FITS and ASCII formats.

        Parameters
        ----------
        filename: str
            Root of the filename where to save the observations.

        """
        self.table.write('out/{}.fits'.format(filename))
        self.table.write('out/{}.txt'.format(filename),
                         format='ascii.fixed_width', delimiter=None)


class ObservationsManagerVirtual(object):
    """Virtual observations manager when there is no observations file given
    as input. In that case we only use the list of bands given in the
    pcigale.ini file.
    """

    def __init__(self, config, **kwargs):
        self.bands = [band for band in config['bands'] if not
                      band.endswith('_err')]

        if len(self.bands) != len(config['bands']):
            print("Warning: error bars were given in the list of bands.")
        elif len(self.bands) == 0:
            print("Warning: no band was given.")

        # We set the other class members to None as they do not make sense in
        # this situation
        self.errors = None
        self.table = None

    def __len__(self):
        """As there is no observed object the length is naturally 0.
        """
        return 0
