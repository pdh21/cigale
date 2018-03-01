# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Copyright (C) 2013-2014 Institute of Astronomy
# Copyright (C) 2014 Yannick Roehlly <yannick@iaora.eu>
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly & Médéric Boquien

from functools import lru_cache

from astropy import log
from astropy.cosmology import WMAP7 as cosmo
import numpy as np
from scipy import optimize
from scipy.constants import parsec
from scipy.special import erf

log.setLevel('ERROR')


def save_chi2(obs, variable, models, chi2, values):
    """Save the chi² and the associated physocal properties

    """
    fname = 'out/{}_{}_chi2-block-{}.npy'.format(obs.id, variable.replace('/',
                                                 '\/'), models.iblock)
    data = np.memmap(fname, dtype=np.float64, mode='w+',
                     shape=(2, chi2.size))
    data[0, :] = chi2
    data[1, :] = values


@lru_cache(maxsize=None)
def compute_corr_dz(model_z, obs_dist):
    """The mass-dependent physical properties are computed assuming the
    redshift of the model. However because we round the observed redshifts to
    two decimals, there can be a difference of 0.005 in redshift between the
    models and the actual observation. At low redshift, this can cause a
    discrepancy in the mass-dependent physical properties: ~0.35 dex at z=0.010
    vs 0.015 for instance. Therefore we correct these physical quantities by
    multiplying them by corr_dz.

    Parameters
    ----------
    model_z: float
        Redshift of the model.
    obs_dist: float
        Luminosity distance of the observed object.

    """
    if model_z == 0.:
        return (obs_dist / (10. * parsec))**2.
    return (obs_dist / cosmo.luminosity_distance(model_z).value)**2.


def dchi2_over_ds2(s, obs_values, obs_errors, mod_values):
    """Function used to estimate the normalization factor in the SED fitting
    process when upper limits are included in the dataset to fit (from Eq. A11
    in Sawicki M. 2012, PASA, 124, 1008).

    Parameters
    ----------
    s: Float
        Contains value onto which we perform minimization = normalization
        factor
    obs_value: RawArray
        Contains observed fluxes for each filter and obseved extensive
        properties.
    obs_errors: RawArray
        Contains observed errors for each filter and extensive properties.
    model_values: RawArray
        Contains modeled fluxes for each filter and extensive properties.
    lim_flag: Boolean
        Tell whether we use upper limits (True) or not (False).

   Returns
    -------
    func: Float
        Eq. A11 in Sawicki M. 2012, PASA, 124, 1008).

    """
    # We enter into this function if lim_flag = True.

    # The mask "data" selects the filter(s) for which measured fluxes are given
    # i.e., when obs_fluxes is >=0. and obs_errors >=0.
    # The mask "lim" selects the filter(s) for which upper limits are given
    # i.e., when obs_errors < 0

    wlim = np.where(np.isfinite(obs_errors) & (obs_errors < 0.))
    wdata = np.where(obs_errors >= 0.)

    mod_values_data = mod_values[wdata]
    mod_values_lim = mod_values[wlim]

    obs_values_data = obs_values[wdata]
    obs_values_lim = obs_values[wlim]

    obs_errors_data = obs_errors[wdata]
    obs_errors_lim = -obs_errors[wlim]

    dchi2_over_ds_data = np.sum(
        (obs_values_data-s*mod_values_data) *
        mod_values_data/(obs_errors_data*obs_errors_data))

    dchi2_over_ds_lim = np.sqrt(2./np.pi)*np.sum(
        mod_values_lim*np.exp(
            -np.square(
                (obs_values_lim-s*mod_values_lim)/(np.sqrt(2)*obs_errors_lim)
                      )
                             )/(
            obs_errors_lim*(
                1.+erf(
                  (obs_values_lim-s*mod_values_lim)/(np.sqrt(2)*obs_errors_lim)
                      )
                           )
                               )
                                                )

    func = dchi2_over_ds_data - dchi2_over_ds_lim

    return func


def _compute_scaling(model_fluxes, model_propsmass, observation):
    """Compute the scaling factor to be applied to the model fluxes to best fit
    the observations. Note that we look over the bands to avoid the creation of
    an array of the same size as the model_fluxes array. Because we loop on the
    bands and not on the models, the impact on the performance should be small.

    Parameters
    ----------
    model_fluxes: array
        Fluxes of the models
    model_propsmass: array
        Extensive properties of the models to be fitted
    observation: Class
        Class instance containing the fluxes, intensive properties, extensive
        properties and their errors, for a sigle observation.
    Returns
    -------
    scaling: array
        Scaling factors minimising the χ²
    """

    obs_fluxes = observation.fluxes
    obs_fluxes_err = observation.fluxes_err
    obs_propsmass = observation.extprops
    obs_propsmass_err = observation.extprops_err

    num = np.zeros(model_fluxes.shape[1])
    denom = np.zeros(model_fluxes.shape[1])
    for i in range(obs_fluxes.size):
        if np.isfinite(obs_fluxes[i]) and obs_fluxes_err[i] > 0.:
            num += model_fluxes[i, :] * (obs_fluxes[i] / (obs_fluxes_err[i] *
                                                          obs_fluxes_err[i]))
            denom += np.square(model_fluxes[i, :] * (1./obs_fluxes_err[i]))
    for i in range(obs_propsmass.size):
        if np.isfinite(obs_propsmass[i]) and obs_propsmass_err[i] > 0.:
            num += model_propsmass[i, :] * (obs_propsmass[i] /
                                            (obs_propsmass_err[i] *
                                             obs_propsmass_err[i]))
            denom += np.square(model_propsmass[i, :] *
                               (1./obs_propsmass_err[i]))

    return num/denom


def compute_chi2(model_fluxes, model_props, model_propsmass, observation,
                 corr_dz, lim_flag):
    """Compute the χ² of observed fluxes with respect to the grid of models. We
    take into account upper limits if need be. Note that we look over the bands
    to avoid the creation of an array of the same size as the model_fluxes
    array. Because we loop on the bands and not on the models, the impact on
    the performance should be small.

    Parameters
    ----------
    model_fluxes: array
        2D grid containing the fluxes of the models
    model_props: array
        2D grid containing the intensive properties of the models
    model_propsmass: array
        2D grid containing the extensive properties of the models
    observation: Class
        Class instance containing the fluxes, intensive properties, extensive
        properties and their errors, for a sigle observation.
    corr_dz: correction factor to scale the extensive properties to the right
        distance
    lim_flag: boolean
        Boolean indicating whether upper limits should be treated (True) or
        discarded (False)

    Returns
    -------
    chi2: array
        χ² for all the models in the grid
    scaling: array
        scaling of the models to obtain the minimum χ²
    """
    limits = lim_flag and np.any(observation.fluxes_err +
                                 observation.extprops_err <= 0.)
    scaling = _compute_scaling(model_fluxes, model_propsmass, observation)

    obs_fluxes = observation.fluxes
    obs_fluxes_err = observation.fluxes_err
    obs_props = observation.intprops
    obs_props_err = observation.intprops_err
    obs_propsmass = observation.extprops
    obs_propsmass_err = observation.extprops_err

    # Some observations may not have flux values in some filter(s), but
    # they can have upper limit(s).
    if limits == True:
        obs_values = np.concatenate((obs_fluxes, obs_propsmass))
        obs_values_err = np.concatenate((obs_fluxes_err, obs_propsmass_err))
        model_values = np.concatenate((model_fluxes, model_propsmass))
        for imod in range(scaling.size):
            scaling[imod] = optimize.root(dchi2_over_ds2, scaling[imod],
                                          args=(obs_values, obs_values_err,
                                                model_values[:, imod])).x

    # χ² of the comparison of each model to each observation.
    chi2 = np.zeros(model_fluxes.shape[1])
    for i in range(obs_fluxes.size):
        if np.isfinite(obs_fluxes[i]) and obs_fluxes_err[i] > 0.:
            chi2 += np.square((obs_fluxes[i] - model_fluxes[i, :] * scaling) *
                              (1./obs_fluxes_err[i]))

    for i in range(obs_propsmass.size):
        if np.isfinite(obs_propsmass[i]):
            chi2 += np.square((obs_propsmass[i] - corr_dz * (scaling *
                               model_propsmass[i, :])) *
                              (1./obs_propsmass_err[i]))

    for i in range(obs_props.size):
        if np.isfinite(obs_props[i]):
            chi2 += np.square((obs_props[i] - model_props[i, :]) *
                              (1./obs_props_err[i]))
    # they can have upper limit(s).
    if limits == True:
        for i, obs_error in enumerate(obs_fluxes_err):
            if obs_error < 0.:
                chi2 -= 2. * np.log(.5 *
                                    (1. + erf(((obs_fluxes[i] -
                                     model_fluxes[i, :] * scaling) /
                                     (-np.sqrt(2.)*obs_fluxes_err[i])))))

    return chi2, scaling


def weighted_param(param, weights):
    """Compute the weighted mean and standard deviation of an array of data.

    Parameters
    ----------
    param: array
        Values of the parameters for the entire grid of models
    weights: array
        Weights by which to weight the parameter values

    Returns
    -------
    mean: float
        Weighted mean of the parameter values
    std: float
        Weighted standard deviation of the parameter values

    """

    mean = np.average(param, weights=weights)
    std = np.sqrt(np.average((param-mean)**2, weights=weights))

    return (mean, std)
