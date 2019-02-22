# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Copyright (C) 2013-2014 Institute of Astronomy
# Copyright (C) 2014 Laboratoire d'Astrophysique de Marseille, AMU
# Copyright (C) 2014 Yannick Roehlly <yannick@iaora.eu>
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly, Médéric Boquien & Denis Burgarella

from copy import deepcopy

import numpy as np

from ..utils import nothread
from .utils import save_chi2, compute_corr_dz, compute_chi2, weighted_param
from ...warehouse import SedWarehouse


def init_sed(models, counter):
    """Initializer of the pool of processes to share variables between workers.

    Parameters
    ----------
    models: ModelsManagers
        Manages the storage of the computed models (fluxes and properties).
    counter: Counter class object
        Counter for the number of models computed

    """
    global gbl_warehouse, gbl_models, gbl_counter

    # Limit the number of threads to 1 if we use MKL in order to limit the
    # oversubscription of the CPU/RAM.
    nothread()

    gbl_warehouse = SedWarehouse()

    gbl_models = models
    gbl_counter = counter


def init_analysis(models, results, counter):
    """Initializer of the pool of processes to share variables between workers.

    Parameters
    ----------
    models: ModelsManagers
        Manages the storage of the computed models (fluxes and properties).
    results: ResultsManager
        Contains the estimates and errors on the properties.
    counter: Counter class object
        Counter for the number of objects analysed

    """
    global gbl_models, gbl_obs, gbl_results, gbl_counter

    gbl_models = models
    gbl_obs = models.obs
    gbl_results = results
    gbl_counter = counter


def init_bestfit(conf, params, observations, results, counter):
    """Initializer of the pool of processes to share variables between workers.

    Parameters
    ----------
    conf: dict
        Contains the pcigale.ini configuration.
    params: ParametersManager
        Manages the parameters from a 1D index.
    observations: astropy.Table
        Contains the observations including the filter names.
    results: ResultsManager
        Contains the estimates and errors on the properties.
    counter: Counter class object
        Counter for the number of objects analysed

    """
    global gbl_warehouse, gbl_conf, gbl_params, gbl_obs
    global gbl_results, gbl_counter

    gbl_warehouse = SedWarehouse()

    gbl_conf = conf
    gbl_params = params
    gbl_obs = observations
    gbl_results = results
    gbl_counter = counter


def sed(idx, midx):
    """Worker process to retrieve a SED and affect the relevant data to an
    instance of ModelsManager.

    Parameters
    ----------
    idx: int
        Index of the model within the current block of models.
    midx: int
        Global index of the model.

    """
    sed = gbl_warehouse.get_sed(gbl_models.params.modules,
                                gbl_models.params.from_index(midx))

    if 'sfh.age' in sed.info and sed.info['sfh.age'] > sed.info['universe.age']:
        for band in gbl_models.flux:
            gbl_models.flux[band][idx] = np.nan
        for prop in gbl_models.extprop:
            gbl_models.extprop[prop][idx] = np.nan
        for prop in gbl_models.intprop:
            gbl_models.intprop[prop][idx] = np.nan
    else:
        for band in gbl_models.flux:
            gbl_models.flux[band][idx] = sed.compute_fnu(band)
        for prop in gbl_models.extprop:
            gbl_models.extprop[prop][idx] = sed.info[prop]
        for prop in gbl_models.intprop:
            gbl_models.intprop[prop][idx] = sed.info[prop]

    gbl_counter.inc()


def analysis(idx, obs):
    """Worker process to analyse the PDF and estimate parameters values and
    store them in an instance of ResultsManager.

    Parameters
    ----------
    idx: int
        Index of the observation. This is necessary to put the computed values
        at the right location in the ResultsManager.
    obs: row
        Input data for an individual object

    """
    np.seterr(invalid='ignore')

    if obs.redshift >= 0.:
        # We pick the the models with the closest redshift using a slice to
        # work on views of the arrays and not on copies to save on RAM.
        z = np.array(
            gbl_models.conf['sed_modules_params']['redshifting']['redshift'])
        wz = slice(np.abs(obs.redshift-z).argmin(), None, z.size)
        corr_dz = compute_corr_dz(z[wz.start], obs)
    else:  # We do not know the redshift so we use the full grid
        wz = slice(0, None, 1)
        corr_dz = 1.

    chi2, scaling = compute_chi2(gbl_models, obs, corr_dz, wz,
                                 gbl_models.conf['analysis_params']['lim_flag'])

    if np.any(chi2 < -np.log(np.finfo(np.float64).tiny) * 2.):
        # We use the exponential probability associated with the χ² as
        # likelihood function.
        likelihood = np.exp(-chi2 / 2.)
        wlikely = np.where(np.isfinite(likelihood))
        # If all the models are valid, it is much more efficient to use a slice
        if likelihood.size == wlikely[0].size:
            wlikely = slice(None, None)
        likelihood = likelihood[wlikely]
        scaling_l = scaling[wlikely]

        gbl_results.bayes.weight[idx] = np.nansum(likelihood)

        # We compute the weighted average and standard deviation using the
        # likelihood as weight.
        for prop in gbl_results.bayes.intmean:
            if prop.endswith('_log'):
                values = gbl_models.intprop[prop[:-4]][wz]
                _ = np.log10
            else:
                values = gbl_models.intprop[prop][wz]
                _ = lambda x: x
            mean, std = weighted_param(_(values[wlikely]), likelihood)
            gbl_results.bayes.intmean[prop][idx] = mean
            gbl_results.bayes.interror[prop][idx] = std
            if gbl_models.conf['analysis_params']['save_chi2'] is True:
                save_chi2(obs, prop, gbl_models, chi2, values)

        for prop in gbl_results.bayes.extmean:
            if prop.endswith('_log'):
                values = gbl_models.extprop[prop[:-4]][wz]
                _ = np.log10
            else:
                values = gbl_models.extprop[prop][wz]
                _ = lambda x: x
            mean, std = weighted_param(_(values[wlikely] * scaling_l * corr_dz),
                                       likelihood)
            gbl_results.bayes.extmean[prop][idx] = mean
            gbl_results.bayes.exterror[prop][idx] = std
            if gbl_models.conf['analysis_params']['save_chi2'] is True:
                save_chi2(obs, prop, gbl_models, chi2,
                          values * scaling * corr_dz)

        for band in gbl_results.bayes.fluxmean:
            values = gbl_models.flux[band][wz]
            mean, std = weighted_param(values[wlikely] * scaling_l,
                                       likelihood)
            gbl_results.bayes.fluxmean[band][idx] = mean
            gbl_results.bayes.fluxerror[band][idx] = std
            if gbl_models.conf['analysis_params']['save_chi2'] is True:
                save_chi2(obs, band, gbl_models, chi2, values * scaling)

        best_idx_z = np.nanargmin(chi2)
        gbl_results.best.chi2[idx] = chi2[best_idx_z]
        gbl_results.best.scaling[idx] = scaling[best_idx_z]
        gbl_results.best.index[idx] = (wz.start + best_idx_z*wz.step +
                                       gbl_models.block.start)
    else:
        # It sometimes happens because models are older than the Universe's age
        print("No suitable model found for the object {}. It may be that "
              "models are older than the Universe or that your chi² are very "
              "large.".format(obs.id))

    gbl_counter.inc()

def bestfit(oidx, obs):
    """Worker process to compute and save the best fit.

    Parameters
    ----------
    oidx: int
        Index of the observation. This is necessary to put the computed values
        at the right location in the ResultsManager.
    obs: row
        Input data for an individual object

    """
    np.seterr(invalid='ignore')

    best_index = int(gbl_results.best.index[oidx])

    # We compute the model at the exact redshift not to have to correct for the
    # difference between the object and the grid redshifts.
    params = deepcopy(gbl_params.from_index(best_index))
    if obs.redshift >= 0.:
        model_z = params[gbl_params.modules.index('redshifting')]['redshift']
        params[gbl_params.modules.index('redshifting')]['redshift'] = obs.redshift
        # Correct fluxes for the fact that the scaling factor was computed on
        # the grid redshift. Because of the difference in redshift the distance
        # is different and must be reflected in the scaling
        corr_scaling = compute_corr_dz(model_z, obs) / \
                       compute_corr_dz(obs.redshift, obs)
    else:  # The model redshift is always exact in redhisfting mode
        corr_scaling = 1.

    sed = gbl_warehouse.get_sed(gbl_params.modules, params)

    # Handle the case where the distance does not correspond to the redshift.
    if obs.redshift >= 0.:
        corr_dz = (obs.distance / sed.info['universe.luminosity_distance']) ** 2
    else:
        corr_dz = 1.

    scaling = gbl_results.best.scaling[oidx] * corr_scaling

    for band in gbl_results.best.flux:
        gbl_results.best.flux[band][oidx] = sed.compute_fnu(band) * scaling

    # If the distance is user defined, the redshift-based luminosity distance
    # of the model is probably incorrect so we replace it
    sed.add_info('universe.luminosity_distance', obs.distance, force=True)
    for prop in gbl_results.best.intprop:
        gbl_results.best.intprop[prop][oidx] = sed.info[prop]

    for prop in gbl_results.best.extprop:
        gbl_results.best.extprop[prop][oidx] = sed.info[prop] * scaling \
                                                   * corr_dz

    if gbl_conf['analysis_params']["save_best_sed"]:
        sed.to_fits('out/{}'.format(obs.id), scaling)

    gbl_counter.inc()
