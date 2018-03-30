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

from astropy.table import Table, Column
import numpy as np

from .utils import SharedArray


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
        nobs = len(models.obs)
        self.propertiesnames = models.propertiesnames
        self.extpropnames = models.massproportional.\
            intersection(models.propertiesnames)
        self.intpropnames = set(models.propertiesnames) - self.extpropnames
        self.nproperties = len(models.propertiesnames)

        # Arrays where we store the data related to the models. For memory
        # efficiency reasons, we use RawArrays that will be passed in argument
        # to the pool. Each worker will fill a part of the RawArrays. It is
        # important that there is no conflict and that two different workers do
        # not write on the same section.
        self.intmean = {prop: SharedArray(nobs) for prop in self.intpropnames}
        self.interror = {prop: SharedArray(nobs) for prop in self.intpropnames}
        self.extmean = {prop: SharedArray(nobs) for prop in self.extpropnames}
        self.exterror = {prop: SharedArray(nobs) for prop in self.extpropnames}
        self.weight = SharedArray(nobs)

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = mean

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, error):
        self._error = error

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = weight

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

        merged = results[0]
        intmean = {prop: np.array([result.intmean[prop].data
                                   for result in results])
                   for prop in merged.intmean}
        interror = {prop: np.array([result.interror[prop].data
                                    for result in results])
                    for prop in merged.interror}
        extmean = {prop: np.array([result.extmean[prop].data
                                   for result in results])
                   for prop in merged.extmean}
        exterror = {prop: np.array([result.exterror[prop].data
                                    for result in results])
                    for prop in merged.exterror}
        weight = np.array([result.weight.data for result in results])

        totweight = np.sum(weight, axis=0)

        for prop in merged.intmean:
            merged.intmean[prop].data[:] = np.sum(
                intmean[prop] * weight, axis=0) / totweight

            # We compute the merged standard deviation by combining the
            # standard deviations for each block. See
            # http://stats.stackexchange.com/a/10445 where the number of
            # datapoints has been substituted with the weights. In short we
            # exploit the fact that Var(X) = E(Var(X)) + Var(E(X)).
            merged.interror[prop].data[:] = np.sqrt(np.sum(
                weight * (interror[prop]**2. + (intmean[prop]-merged.intmean[prop].data)**2), axis=0) / totweight)

        for prop in merged.extmean:
            merged.extmean[prop].data[:] = np.sum(
                extmean[prop] * weight, axis=0) / totweight

            # We compute the merged standard deviation by combining the
            # standard deviations for each block. See
            # http://stats.stackexchange.com/a/10445 where the number of
            # datapoints has been substituted with the weights. In short we
            # exploit the fact that Var(X) = E(Var(X)) + Var(E(X)).
            merged.exterror[prop].data[:] = np.sqrt(np.sum(
                weight * (exterror[prop]**2. + (extmean[prop]-merged.extmean[prop].data)**2), axis=0) / totweight)

        for prop in merged.extmean:
            if prop.endswith('_log'):
                merged.exterror[prop].data[:] = \
                    np.maximum(0.02, merged.exterror[prop].data)
            else:
                merged.exterror[prop].data[:] = \
                    np.maximum(0.05 * merged.extmean[prop].data,
                               merged.exterror[prop].data)

        return merged


class BestResultsManager(object):
    """This class contains the physical properties of the best fit of the
    analysed objects. It is constructed from a ModelsManager instance, which
    provides the required information on the shape of the arrays. Because it
    can contain the partial results for only one block of models, we also store
    the index so we can merge different instances to compute the best fit.

    """

    def __init__(self, models):
        self.obs = models.obs
        nobs = len(models.obs)
        # Arrays where we store the data related to the models. For memory
        # efficiency reasons, we use RawArrays that will be passed in argument
        # to the pool. Each worker will fill a part of the RawArrays. It is
        # important that there is no conflict and that two different workers do
        # not write on the same section.
        self.flux = {band: SharedArray(nobs) for band in models.obs.bands}
        allintpropnames = set(models.allpropertiesnames) - models.massproportional
        allextpropnames = set(models.allpropertiesnames) - allintpropnames
        self.intprop = {prop: SharedArray(nobs)
                        for prop in allintpropnames}
        self.extprop = {prop: SharedArray(nobs)
                        for prop in allextpropnames}
        self.chi2 = SharedArray(nobs)
        self.scaling = SharedArray(nobs)

        # We store the index as a float to work around python issue #10746
        self.index = SharedArray(nobs)

    @property
    def flux(self):
        """Returns a shared array containing the fluxes of the best fit for
        each observation.

        """
        return self._flux

    @flux.setter
    def flux(self, flux):
        self._flux = flux

    @property
    def intprop(self):
        """Returns a shared array containing the fluxes of the best fit for
        each observation.

        """
        return self._intprop

    @intprop.setter
    def intprop(self, intprop):
        self._intprop = intprop

    @property
    def extprop(self):
        """Returns a shared array containing the fluxes of the best fit for
        each observation.

        """
        return self._extprop

    @extprop.setter
    def extprop(self, extprop):
        self._extprop = extprop

    @property
    def chi2(self):
        """Returns a shared array containing the raw chi² of the best fit for
        each observation.

        """
        return self._chi2.data

    @chi2.setter
    def chi2(self, chi2):
        self._chi2 = chi2

    @property
    def index(self):
        """Returns a shared array containing the index of the best fit for each
        observation.

        """
        return self._index.data

    @index.setter
    def index(self, index):
        self._index = index

    @property
    def scaling(self):
        """Returns a shared array containing the scaling of the best fit for each
        observation.

        """
        return self._scaling.data

    @scaling.setter
    def scaling(self, scaling):
        self._scaling = scaling

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

        best = np.argmin([result.chi2 for result in results], axis=0)

        merged = results[0]
        for iobs, bestidx in enumerate(best):
            for band in merged.flux:
                merged.flux[band].data[iobs] = \
                    results[bestidx].flux[band].data[iobs]
            for prop in merged.intprop:
                merged.intprop[prop].data[iobs] = \
                    results[bestidx].intprop[prop].data[iobs]
            for prop in merged.extprop:
                merged.extprop[prop].data[iobs] = \
                    results[bestidx].extprop[prop].data[iobs]
            merged.chi2[iobs] = results[bestidx].chi2[iobs]
            merged.scaling[iobs] = results[bestidx].scaling[iobs]
            merged.index[iobs] = results[bestidx].index[iobs]

        return merged

    def analyse_chi2(self):
        """Function to analyse the best chi^2 and find out what fraction of
         objects seems to be overconstrainted.

        """
        obs = [self.obs.table[obs].data for obs in self.obs.tofit]
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

        for prop in sorted(self.bayes.intmean):
            table.add_column(Column(self.bayes.intmean[prop].data,
                                    name="bayes."+prop))
            table.add_column(Column(self.bayes.interror[prop].data,
                                    name="bayes."+prop+"_err"))
        for prop in sorted(self.bayes.extmean):
            table.add_column(Column(self.bayes.extmean[prop].data,
                                    name="bayes."+prop))
            table.add_column(Column(self.bayes.exterror[prop].data,
                                    name="bayes."+prop+"_err"))

        table.add_column(Column(self.best.chi2, name="best.chi_square"))
        obs = [self.obs.table[obs].data for obs in self.obs.tofit]
        nobs = np.count_nonzero(np.isfinite(obs), axis=0)
        table.add_column(Column(self.best.chi2 / (nobs - 1),
                                name="best.reduced_chi_square"))

        for prop in sorted(self.best.intprop):
            table.add_column(Column(self.best.intprop[prop].data,
                                    name="best."+prop))
        for prop in sorted(self.best.extprop):
            table.add_column(Column(self.best.extprop[prop].data,
                                    name="best."+prop))

        for band in self.obs.bands:
            table.add_column(Column(self.best.flux[band].data,
                                    name="best."+band, unit='mJy'))


        table.write("out/{}.txt".format(filename), format='ascii.fixed_width',
                    delimiter=None)
        table.write("out/{}.fits".format(filename), format='fits')
