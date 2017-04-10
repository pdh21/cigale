# -*- coding: utf-8 -*-
# Copyright (C) 2017 Universidad de Antofagasta
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Médéric Boquien

"""These classes manage the results from the analysis. The main class
ResultsManager contains instances of BayesResultsManager and BestResultsManager
that contain the bayesian and best-fit estimates of the physical properties
along with the names of the parameters, which are proportional to the mass,
etc. Each of these classes contain a merge() method that allows to combine
results of the analysis with different blocks of models.
"""

import ctypes
from multiprocessing.sharedctypes import RawArray

from astropy.table import Table, Column
import numpy as np


def shared_array(shape):
    """Create a shared array that can be read/written by parallel processes
    """
    return RawArray(ctypes.c_double, int(np.product(shape)))


class BayesResultsManager(object):
    """This class contains the results of the bayesian estimates of the
    physical properties of the analysed objects. It is constructed from a
    ModelsManager instance, which provides the required information on the
    shape of the arrays. Because it can contain the partial results for only
    one block of models, we also store the sum of the model weights (that is
    the likelihood) so we can merge different instances to compute the combined
    estimates of the physical properties.

    """
    def __init__(self, models):
        self.nobs = len(models.obs)
        self.propertiesnames = models.propertiesnames
        self.massproportional = models.massproportional.\
                                   intersection(models.propertiesnames)
        self.nproperties = len(models.propertiesnames)

        # Arrays where we store the data related to the models. For memory
        # efficiency reasons, we use RawArrays that will be passed in argument
        # to the pool. Each worker will fill a part of the RawArrays. It is
        # important that there is no conflict and that two different workers do
        # not write on the same section.
        self._means = shared_array((self.nobs, self.nproperties))
        self._errors = shared_array((self.nobs, self.nproperties))
        self._weights = shared_array((self.nobs))

    @staticmethod
    def merge(results):
        """Merge a list of partial results computed on individual blocks of
        models.

        Parameters
        ----------
        results: list of BayesResultsManager instances
            List of the partial results to be merged

        """
        if not (isinstance(results, list) and
                all(isinstance(result, BayesResultsManager)
                    for result in results)):
            raise TypeError("A list of BayesResultsManager is required.")

        if len(results) == 1:
            results[0]._weights = None
            return results[0]

        means = np.array([result.means for result in results])
        errors = np.array([result.errors for result in results])
        weights = np.array([result.weights for result in results])[..., None]

        merged = results[0]
        merged._means = shared_array((merged.nobs, merged.nproperties))
        merged._errors = shared_array((merged.nobs, merged.nproperties))
        merged._weights = None

        sumweights = np.sum(weights, axis=0)

        merged.means[:] = np.sum(means * weights, axis=0) / sumweights

        # We compute the merged standard deviation by combining the standard
        # deviations for each block. See http://stats.stackexchange.com/a/10445
        # where the number of datapoints has been substituted with the weights.
        # In short we exploit the fact that Var(X) = E(Var(X)) + Var(E(X)).
        merged.errors[:] = np.sqrt(np.sum(weights * (errors**2 +
                                          (means-merged.means)**2), axis=0) /
                                   sumweights)

        return merged

    @property
    def means(self):
        """Returns a shared array containing the weighted means for each
        physical property and each observation.

        """
        return np.ctypeslib.as_array(self._means).reshape((self.nobs,
                                                           self.nproperties))

    @property
    def errors(self):
        """Returns a shared array containing the weighted standard deviation
         for each physical property and each observation.

        """
        return np.ctypeslib.as_array(self._errors).reshape((self.nobs,
                                                            self.nproperties))

    @property
    def weights(self):
        """Returns a shared array containing the some of the weights for each
        each observation.

        """
        return np.ctypeslib.as_array(self._weights)


class BestResultsManager(object):
    """This class contains the physical properties of the best fit of the
    analysed objects. It is constructed from a ModelsManager instance, which
    provides the required information on the shape of the arrays. Because it
    can contain the partial results for only one block of models, we also store
    the index so we can merge different instances to compute the best fit.

    """

    def __init__(self, models):
        self.obs = models.obs
        self.nbands = len(models.obs.bands)
        self.nobs = len(models.obs)
        self.propertiesnames = models.allpropertiesnames
        self.massproportional = models.massproportional
        self.nproperties = len(models.allpropertiesnames)

        self._fluxes_shape = (self.nobs, self.nbands)
        self._properties_shape = (self.nobs, self.nproperties)

        # Arrays where we store the data related to the models. For memory
        # efficiency reasons, we use RawArrays that will be passed in argument
        # to the pool. Each worker will fill a part of the RawArrays. It is
        # important that there is no conflict and that two different workers do
        # not write on the same section.
        self._fluxes = shared_array(self._fluxes_shape)
        self._properties = shared_array(self._properties_shape)
        self._chi2 = shared_array(self.nobs)
        # We store the index as a float to work around python issue #10746
        self._index = shared_array(self.nobs)

    @property
    def fluxes(self):
        """Returns a shared array containing the fluxes of the best fit for
        each observation.

        """
        return np.ctypeslib.as_array(self._fluxes).reshape(self._fluxes_shape)

    @property
    def properties(self):
        """Returns a shared array containing the physical properties of the
        best fit for each observation.

        """
        return np.ctypeslib.as_array(self._properties)\
                   .reshape(self._properties_shape)

    @property
    def chi2(self):
        """Returns a shared array containing the raw chi² of the best fit for
        each observation.

        """
        return np.ctypeslib.as_array(self._chi2)

    @property
    def index(self):
        """Returns a shared array containing the index of the best fit for each
        observation.

        """
        return np.ctypeslib.as_array(self._index)

    @staticmethod
    def merge(results):
        """Merge a list of partial results computed on individual blocks of
        models.

        Parameters
        ----------
        results: list of BestResultsManager instances
            List of the partial results to be merged

        """
        if not (isinstance(results, list) and
                all(isinstance(result, BestResultsManager)
                    for result in results)):
            raise TypeError("A list of BestResultsManager is required.")

        if len(results) == 1:
            return results[0]

        fluxes = np.array([result.fluxes for result in results])
        properties = np.array([result.properties for result in results])
        chi2 = np.array([result.chi2 for result in results])
        index = np.array([result.index for result in results])

        merged = results[0]
        merged._fluxes = shared_array((merged.nobs, merged.nbands))
        merged._properties = shared_array((merged.nobs, merged.nproperties))
        merged._chi2 = shared_array(merged.nobs)
        # We store the index as a float to work around python issue #10746
        merged._index = shared_array(merged.nobs)

        for iobs, bestidx in enumerate(np.argmin(chi2, axis=0)):
            merged.fluxes[iobs, :] = fluxes[bestidx, iobs, :]
            merged.properties[iobs, :] = properties[bestidx, iobs, :]
            merged.chi2[iobs] = chi2[bestidx, iobs]
            merged.index[iobs] = index[bestidx, iobs]

        return merged

    def analyse_chi2(self):
        """Function to analyse the best chi^2 and find out what fraction of
         objects seems to be overconstrainted.

        """
        obs = [self.obs.table[obs].data for obs in self.obs.bands]
        nobs = np.count_nonzero(np.isfinite(obs), axis=0)
        chi2_red = self.chi2 / (nobs - 1)
        # If low values of reduced chi^2, it means that the data are overfitted
        # Errors might be under-estimated or not enough valid data.
        print("\n{}% of the objects have chi^2_red~0 and {}% chi^2_red<0.5"
              .format(np.round((chi2_red < 1e-12).sum() / chi2_red.size, 1),
                      np.round((chi2_red < 0.5).sum() / chi2_red.size, 1)))


class ResultsManager(object):
    """This class contains the physical properties (best fit and bayesian) of
    the analysed objects. It is constructed from a ModelsManager instance,
    which provides the required information to initialise the instances of
    BestResultsManager and BayesResultsManager that store the results.

    """

    def __init__(self, models):
        self.conf = models.conf
        self.obs = models.obs
        self.params = models.params

        self.bayes = BayesResultsManager(models)
        self.best = BestResultsManager(models)

    @staticmethod
    def merge(results):
        """Merge a list of partial results computed on individual blocks of
        models.

        Parameters
        ----------
        results: list of ResultsManager instances
            List of the partial results to be merged

        """
        merged = results[0]
        merged.bayes = BayesResultsManager.merge([result.bayes
                                                  for result in results])
        merged.best = BestResultsManager.merge([result.best
                                                for result in results])

        return merged

    def save(self, filename):
        """Save the estimated values derived from the analysis of the PDF and
        the parameters associated with the best fit. A simple text file and a
        FITS file are generated.

        Parameters
        ----------
        filename:

        """
        table = Table()

        table.add_column(Column(self.obs.table['id'], name="id"))

        for idx, name in enumerate(self.bayes.propertiesnames):
            table.add_column(Column(self.bayes.means[:, idx],
                                    name="bayes."+name))
            table.add_column(Column(self.bayes.errors[:, idx],
                                    name="bayes."+name+"_err"))

        table.add_column(Column(self.best.chi2, name="best.chi_square"))
        obs = [self.obs.table[obs].data for obs in self.obs.bands]
        nobs = np.count_nonzero(np.isfinite(obs), axis=0)
        table.add_column(Column(self.best.chi2 / (nobs - 1),
                                name="best.reduced_chi_square"))

        for idx, name in enumerate(self.best.propertiesnames):
            table.add_column(Column(self.best.properties[:, idx],
                                    name="best."+name))

        for idx, name in enumerate(self.obs.bands):
            table.add_column(Column(self.best.fluxes[:, idx],
                                    name="best."+name, unit='mJy'))

        table.write("out/{}.txt".format(filename), format='ascii.fixed_width',
                    delimiter=None)
        table.write("out/{}.fits".format(filename), format='fits')
