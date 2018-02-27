# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Copyright (C) 2013-2014 Institute of Astronomy
# Copyright (C) 2014 Laboratoire d'Astrophysique de Marseille, AMU
# Copyright (C) 2014 Yannick Roehlly <yannick@iaora.eu>
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly, Médéric Boquien & Denis Burgarella

from copy import deepcopy
import time

import numpy as np

from ..utils import nothread
from .utils import save_chi2, compute_corr_dz, compute_chi2, weighted_param
from ...warehouse import SedWarehouse


def init_sed(models, t0, ncomputed):
    """Initializer of the pool of processes to share variables between workers.

    Parameters
    ----------
    models: ModelsManagers
        Manages the storage of the computed models (fluxes and properties).
    t0: float
        Time of the beginning of the computation.
    ncomputed: Value
        Number of computed models. Shared among workers.

    """
    global gbl_previous_idx, gbl_warehouse, gbl_models, gbl_obs
    global gbl_properties, gbl_t0, gbl_ncomputed

    # Limit the number of threads to 1 if we use MKL in order to limit the
    # oversubscription of the CPU/RAM.
    nothread()

    gbl_previous_idx = -1
    gbl_warehouse = SedWarehouse()

    gbl_models = models
    gbl_obs = models.obs
    gbl_properties = [prop[:-4] if prop.endswith('_log') else prop for prop in
                      models.propertiesnames]
    gbl_t0 = t0
    gbl_ncomputed = ncomputed


def init_analysis(models, results, t0, ncomputed):
    """Initializer of the pool of processes to share variables between workers.

    Parameters
    ----------
    models: ModelsManagers
        Manages the storage of the computed models (fluxes and properties).
    results: ResultsManager
        Contains the estimates and errors on the properties.
    t0: float
        Time of the beginning of the computation.
    ncomputed: Value
        Number of computed models. Shared among workers.

    """
    global gbl_models, gbl_obs, gbl_results, gbl_t0, gbl_ncomputed

    gbl_models = models
    gbl_obs = models.obs
    gbl_results = results
    gbl_t0 = t0
    gbl_ncomputed = ncomputed


def init_bestfit(conf, params, observations, results, t0, ncomputed):
    """Initializer of the pool of processes to share variables between workers.

    Parameters
    ----------
    conf: dict
        Contains the pcigale.ini configuration.
    params: ParametersManager
        Manages the parameters from a 1D index.
    observations: astropy.Table
        Contains the observations including the filter names.
    ncomputed: Value
        Number of computed models. Shared among workers.
    t0: float
        Time of the beginning of the computation.
    results: ResultsManager
        Contains the estimates and errors on the properties.
    offset: integer
        Offset of the block to retrieve the global model index.

    """
    global gbl_previous_idx, gbl_warehouse, gbl_conf, gbl_params, gbl_obs
    global gbl_results, gbl_t0, gbl_ncomputed

    gbl_previous_idx = -1
    gbl_warehouse = SedWarehouse()

    gbl_conf = conf
    gbl_params = params
    gbl_obs = observations
    gbl_results = results
    gbl_t0 = t0
    gbl_ncomputed = ncomputed


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
    global gbl_previous_idx
    if gbl_previous_idx > -1:
        gbl_warehouse.partial_clear_cache(
            gbl_models.params.index_module_changed(gbl_previous_idx, midx))
    gbl_previous_idx = midx
    sed = gbl_warehouse.get_sed(gbl_models.params.modules,
                                gbl_models.params.from_index(midx))

    if 'sfh.age' in sed.info and sed.info['sfh.age'] > sed.info['universe.age']:
        gbl_models.fluxes[:, idx] = np.full(len(gbl_obs.bands), np.nan)
        gbl_models.properties[:, idx] = np.full(len(gbl_properties), np.nan)
        gbl_models.intprops[:, idx] = np.full(len(gbl_obs.intprops), np.nan)
        gbl_models.extprops[:, idx] = np.full(len(gbl_obs.extprops), np.nan)

    else:
        gbl_models.fluxes[:, idx] = [sed.compute_fnu(filter_)
                                     for filter_ in gbl_obs.bands]
        gbl_models.properties[:, idx] = [sed.info[name]
                                         for name in gbl_properties]
        gbl_models.intprops[:, idx] = [sed.info[name]
                                       for name in gbl_obs.intprops]
        gbl_models.extprops[:, idx] = [sed.info[name]
                                       for name in gbl_obs.extprops]
    with gbl_ncomputed.get_lock():
        gbl_ncomputed.value += 1
        ncomputed = gbl_ncomputed.value
    nmodels = len(gbl_models.block)
    if ncomputed % 250 == 0 or ncomputed == nmodels:
        dt = time.time() - gbl_t0
        print("{}/{} models computed in {:.1f} seconds ({:.1f} models/s)".
              format(ncomputed, nmodels, dt, ncomputed/dt),
              end="\n" if ncomputed == nmodels else "\r")


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
        corr_dz = compute_corr_dz(z[wz.start], obs.redshift)
    else:  # We do not know the redshift so we use the full grid
        wz = slice(0, None, 1)
        corr_dz = 1.

    observation = gbl_obs.observations[idx]
    chi2, scaling = compute_chi2(gbl_models.fluxes[:, wz],
                                 gbl_models.intprops[:, wz],
                                 gbl_models.extprops[:, wz], observation,
                                 gbl_models.conf['analysis_params']['lim_flag'])

    if np.any(np.isfinite(chi2)):
        # We use the exponential probability associated with the χ² as
        # likelihood function.
        likelihood = np.exp(-chi2 / 2.)
        gbl_results.bayes.weights[idx] = np.nansum(likelihood)

        # We compute the weighted average and standard deviation using the
        # likelihood as weight.
        for i, variable in enumerate(gbl_results.bayes.propertiesnames):
            if variable.endswith('_log'):
                variable = variable[:-4]
                _ = np.log10
            else:
                _ = lambda x: x

            if variable in gbl_results.bayes.massproportional:
                values = _(gbl_models.properties[i, wz] * scaling * corr_dz)
            else:
                values = _(gbl_models.properties[i, wz])
                
            wlikely = np.isfinite(likelihood)
            mean, std = weighted_param(values[wlikely], likelihood[wlikely])
            gbl_results.bayes.means[idx, i] = mean
            gbl_results.bayes.errors[idx, i] = std
            if gbl_models.conf['analysis_params']['save_chi2'] is True:
                save_chi2(obs, variable, gbl_models, chi2, values)
        best_idx_z = np.nanargmin(chi2)
        gbl_results.best.chi2[idx] = chi2[best_idx_z]
        gbl_results.best.index[idx] = (wz.start + best_idx_z*wz.step +
                                       gbl_models.block.start)
    else:
        # It sometimes happens because models are older than the Universe's age
        print("No suitable model found for the object {}. One possible origin "
              "is that models are older than the Universe.".format(obs.id))

    with gbl_ncomputed.get_lock():
        gbl_ncomputed.value += 1
        ncomputed = gbl_ncomputed.value
    dt = time.time() - gbl_t0
    print("{}/{} objects analysed in {:.1f} seconds ({:.2f} objects/s)".
          format(ncomputed, len(gbl_models.obs), dt, ncomputed/dt),
          end="\n" if ncomputed == len(gbl_models.obs) else "\r")


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
    global gbl_previous_idx
    if gbl_previous_idx > -1:
        gbl_warehouse.partial_clear_cache(
            gbl_params.index_module_changed(gbl_previous_idx, best_index))
    gbl_previous_idx = best_index

    # We compute the model at the exact redshift not to have to correct for the
    # difference between the object and the grid redshifts.
    params = deepcopy(gbl_params.from_index(best_index))
    params[-1]['redshift'] = obs.redshift
    sed = gbl_warehouse.get_sed(gbl_params.modules, params)

    fluxes = np.array([sed.compute_fnu(filt) for filt in gbl_obs.bands])
    intprops = np.array([sed.info[prop] for prop in gbl_obs.intprops])
    extprops = np.array([sed.info[prop] for prop in gbl_obs.extprops])

    _, scaling = compute_chi2(fluxes[:, None], intprops[:, None],
                              extprops[:, None],  obs,
                              gbl_conf['analysis_params']['lim_flag'])

    gbl_results.best.properties[oidx, :] = [sed.info[k] for k in
                                            gbl_results.best.propertiesnames]
    iprop = [i for i, k in enumerate(gbl_results.best.propertiesnames)
             if k in gbl_results.best.massproportional]
    gbl_results.best.properties[oidx, iprop] *= scaling
    gbl_results.best.fluxes[oidx, :] = fluxes * scaling

    if gbl_conf['analysis_params']["save_best_sed"]:
        sed.to_fits('out/{}'.format(obs['id']), scaling)

    with gbl_ncomputed.get_lock():
        gbl_ncomputed.value += 1
        ncomputed = gbl_ncomputed.value
    dt = time.time() - gbl_t0
    print("{}/{} best fit spectra computed in {:.1f} seconds ({:.2f} objects/s)".
          format(ncomputed, len(gbl_obs), dt, ncomputed/dt), end="\n" if
          ncomputed == len(gbl_obs) else "\r")
