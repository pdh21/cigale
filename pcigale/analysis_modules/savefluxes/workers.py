# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Copyright (C) 2013-2014 Institute of Astronomy
# Copyright (C) 2014 Yannick Roehlly <yannick@iaora.eu>
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly & Médéric Boquien

import numpy as np

from ...warehouse import SedWarehouse
from ..utils import nothread


def init_fluxes(models, counter):
    """Initializer of the pool of processes. It is mostly used to convert
    RawArrays into numpy arrays. The latter are defined as global variables to
    be accessible from the workers.

    Parameters
    ----------
    models: ModelsManagers
        Manages the storage of the computed models (fluxes and properties).
    counter: Counter class object
        Counter for the number of models computed

    """
    global gbl_previous_idx, gbl_warehouse, gbl_models, gbl_obs, gbl_save
    global gbl_counter


    # Limit the number of threads to 1 if we use MKL in order to limit the
    # oversubscription of the CPU/RAM.
    nothread()

    gbl_previous_idx = -1
    gbl_warehouse = SedWarehouse()

    gbl_models = models
    gbl_obs = models.obs
    gbl_save = models.conf['analysis_params']['save_sed']
    gbl_counter = counter


def fluxes(idx, midx):
    """Worker process to retrieve a SED and affect the relevant data to shared
    RawArrays.

    Parameters
    ----------
    idx: int
        Index of the model within the current block of models.

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

    if gbl_save is True:
        sed.to_fits("out/{}".format(midx))

    gbl_counter.inc()
