# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Copyright (C) 2013-2014 Institute of Astronomy
# Copyright (C) 2014 Yannick Roehlly <yannick@iaora.eu>
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly & Médéric Boquien

import time

import numpy as np

from ...warehouse import SedWarehouse
from ..utils import nothread


def init_fluxes(models, t0, ncomputed):
    """Initializer of the pool of processes. It is mostly used to convert
    RawArrays into numpy arrays. The latter are defined as global variables to
    be accessible from the workers.

    Parameters
    ----------
    models: ModelsManagers
        Manages the storage of the computed models (fluxes and properties).
    t0: float
        Time of the beginning of the computation.
    ncomputed: Value
        Number of computed models. Shared among workers.

    """
    global gbl_previous_idx, gbl_warehouse, gbl_models, gbl_obs, gbl_save
    global gbl_t0, gbl_ncomputed


    # Limit the number of threads to 1 if we use MKL in order to limit the
    # oversubscription of the CPU/RAM.
    nothread()

    gbl_previous_idx = -1
    gbl_warehouse = SedWarehouse()

    gbl_models = models
    gbl_obs = models.obs
    gbl_save = models.conf['analysis_params']['save_sed']
    gbl_t0 = t0
    gbl_ncomputed = ncomputed


def fluxes(idx, midx):
    """Worker process to retrieve a SED and affect the relevant data to shared
    RawArrays.

    Parameters
    ----------
    idx: int
        Index of the model within the current block of models.

    """
    global gbl_previous_idx
    if gbl_previous_idx > -1:
        gbl_warehouse.partial_clear_cache(
            gbl_models.params.index_module_changed(gbl_previous_idx, midx))
    gbl_previous_idx = midx

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
        for band in gbl_models.flux.keys():
            gbl_models.flux[band][idx] = sed.compute_fnu(band)
        for prop in gbl_models.extprop.keys():
            gbl_models.extprop[prop][idx] = sed.info[prop]
        for prop in gbl_models.intprop.keys():
            gbl_models.intprop[prop][idx] = sed.info[prop]

    if gbl_save is True:
        sed.to_fits("out/{}".format(midx))

    with gbl_ncomputed.get_lock():
        gbl_ncomputed.value += 1
        ncomputed = gbl_ncomputed.value
    nmodels = len(gbl_models.block)
    if ncomputed % 250 == 0 or ncomputed == nmodels:
        dt = time.time() - gbl_t0
        print("{}/{} models computed in {:.1f} seconds ({:.1f} models/s)".
              format(ncomputed, nmodels, dt, ncomputed/dt),
              end="\n" if ncomputed == nmodels else "\r")
