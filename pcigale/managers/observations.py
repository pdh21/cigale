# -*- coding: utf-8 -*-
# Copyright (C) 2017 Universidad de Antofagasta
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Médéric Boquien

from astropy.table import Column
import numpy as np

from ..utils import read_table
from .utils import get_info


class ObservationsManager(object):
    """Class to abstract the handling of the observations and provide a
    consistent interface for the rest of cigale to deal with observations.

    An ObservationsManager is in charge of reading the input file to memory,
    check the consistency of the data, replace invalid values with NaN, etc.

    """
    def __new__(cls, config, params=None, **kwargs):
        if config['data_file']:
            return ObservationsManagerPassbands(config, params, **kwargs)
        else:
            return ObservationsManagerVirtual(config, **kwargs)


class Observation(object):
    """Class to take one row of the observations table and extract the list of
    fluxes, intensive properties, extensive properties and their errors, that
    are going to be considered in the fit.
    """
    def __init__(self, row, cls):
        self.redshift = row['redshift']
        self.id = row['id']
        self.fluxes = np.array([row[band] for band in cls.bands])
        self.fluxes_err = np.array([row[band + '_err'] for band in cls.bands])
        self.intprops = np.array([row[prop] for prop in cls.intprops])
        self.intprops_err = np.array([row[prop + '_err'] for prop in
                                      cls.intprops])
        self.extprops = np.array([row[prop] for prop in cls.extprops])
        self.extprops_err = np.array([row[prop + '_err'] for prop in
                                      cls.extprops])


class ObservationsManagerPassbands(object):
    """Class to generate a manager for data files providing fluxes in
    passbands.

    A class instance can be used as a sequence. In that case a row is returned
    at each iteration.
    """

    def __init__(self, config, params, defaulterror=0.1, modelerror=0.1,
                 threshold=-9990.):

        self.conf = config
        self.params = params
        self.allpropertiesnames, self.massproportional = get_info(self)
        self.table = read_table(config['data_file'])
        self.bands = [band for band in config['bands'] if not
                      band.endswith('_err')]
        self.bands_err = [band for band in config['bands'] if
                          band.endswith('_err')]
        self.intprops = [prop for prop in config['properties'] if (prop not in
                         self.massproportional and not prop.endswith('_err'))]
        self.intprops_err = [prop for prop in config['properties'] if
                             (prop.endswith('_err') and prop[:-4] not in
                             self.massproportional)]
        self.extprops = [prop for prop in config['properties'] if (prop in
                         self.massproportional and not prop.endswith('_err'))]
        self.extprops_err = [prop for prop in config['properties'] if
                             (prop.endswith('_err') and prop[:-4] in
                             self.massproportional)]
        self.tofit = self.bands + self.intprops + self.extprops
        self.tofit_err = self.bands_err + self.intprops_err + self.extprops_err

        # Sanitise the input
        self._check_filters()
        self._check_errors(defaulterror)
        self._check_invalid(config['analysis_params']['lim_flag'],
                            threshold)
        self._add_model_error(modelerror)

        self.observations = list([Observation(row, self) for row in self.table])

    def __len__(self):
        return len(self.observations)

    def __iter__(self):
        self.idx = 0
        self.max = len(self.observations)

        return self

    def __next__(self):
        if self.idx < self.max:
            obs = self.observations[self.idx]
            self.idx += 1
            return obs
        raise StopIteration

    def _check_filters(self):
        """Check whether the list of filters and poperties makes sense.

        Two situations are checked:
        * If a filter or property to be included in the fit is missing from
        the data file, an exception is raised.
        * If a filter or property is given in the input file but is not to be
        included in the fit, a warning is displayed

        """
        for item in self.tofit + self.tofit_err:
            if item not in self.table.colnames:
                raise Exception("{} to be taken in the fit but not present "
                                "in the observation table.".format(item))

        for item in self.table.colnames:
            if (item != 'id' and item != 'redshift' and item not in self.tofit +
                self.tofit_err):
                self.table.remove_column(item)
                print("Warning: {} in the input file but not to be taken into"
                      " account in the fit.".format(item))

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

        for item in self.tofit:
            error = item + '_err'
            if item in self.intprops:
                if error not in self.intprops_err or error not in self.table.colnames:
                    raise ValueError("Intensive properties errors must be in input file.")
            elif error not in self.tofit_err or error not in self.table.colnames:
                colerr = Column(data=np.fabs(self.table[item] * defaulterror),
                                name=error)
                self.table.add_column(colerr,
                                      index=self.table.colnames.index(item)+1)
                print("Warning: {}% of {} taken as errors.".format(defaulterror *
                                                                   100.,
                                                                   item))

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

        for item in self.bands + self.extprops:
            error = item + '_err'
            w = np.where((self.table[item] < threshold) |
                         (self.table[error] < threshold))
            self.table[item][w] = np.nan
            self.table[error][w] = np.nan
            if upperlimits is False:
                w = np.where(self.table[error] <= 0.)
                self.table[item][w] = np.nan
            else:
                w = np.where(self.table[error] == 0.)
                self.table[item][w] = np.nan
            if np.all(~np.isfinite(self.table[item])):
                allinvalid.append(item)

        for item in allinvalid:
            if item in self.bands:
                self.bands.remove(item)
                self.bands_err.remove(item + '_err')
            elif item in self.extprops:
                self.extprops.remove(item)
                self.extprops_err.remove(item + '_err')
            self.table.remove_columns([item, item + '_err'])
            print("Warning: {} removed as no valid data was found.".format(
                allinvalid))

    def _add_model_error(self, modelerror=0.1):
        """Add in quadrature the error of the model to the input error.

        Parameters
        ----------
        modelerror: float
            Relative error of the models relative to the flux (or property). By
            default 10%.

        """
        if modelerror < 0.:
            raise ValueError("The relative model error must be positive.")

        for item in self.bands + self.extprops:
            error = item + '_err'
            w = np.where(self.table[error] >= 0.)
            self.table[error][w] = np.sqrt(self.table[error][w]**2. + (
                self.table[item][w]*modelerror)**2.)

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
            err = band + '_err'
            self.table[band] = np.random.normal(fits.best.fluxes[:, idx],
                                                np.fabs(self.table[err]))
        for idx, prop in enumerate(self.intprops):
            err = prop + '_err'
            self.table[prop] = np.random.normal(fits.best.intprops[:, idx],
                                                np.fabs(self.table[err]))
        for idx, prop in enumerate(self.extprops):
            err = prop + '_err'
            self.table[prop] = np.random.normal(fits.best.extprops[:, idx],
                                                np.fabs(self.table[err]))

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
        self.bands_err = None
        self.table = None

    def __len__(self):
        """As there is no observed object the length is naturally 0.
        """
        return 0
