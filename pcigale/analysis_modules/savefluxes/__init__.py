# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly

"""
Save fluxes analysis module
===========================

This module does not perform a statistical analysis. It computes and save the
fluxes in a set of filters for all the possible combinations of input SED
parameters.

The data file is used only to get the list of fluxes to be computed.

"""

from collections import OrderedDict
import ctypes
import multiprocessing as mp
import time

from .. import AnalysisModule
from .workers import init_fluxes as init_worker_fluxes
from .workers import fluxes as worker_fluxes
from ...managers.models import ModelsManager
from ...managers.observations import ObservationsManager
from ...managers.parameters import ParametersManager


class SaveFluxes(AnalysisModule):
    """Save fluxes analysis module

    This module saves a table containing all the parameters and desired fluxes
    for all the computed models.

    """

    parameter_list = OrderedDict([
        ("variables", (
            "cigale_string_list()",
            "List of the physical properties to save. Leave empty to save all "
            "the physical properties (not recommended when there are many "
            "models).",
            None
        )),
        ("save_sed", (
            "boolean()",
            "If True, save the generated spectrum for each model.",
            False
        ))
    ])

    @staticmethod
    def _compute_models(conf, observations, params):
        models = ModelsManager(conf, observations, params)

        initargs = (models, time.time(), mp.Value('i', 0))
        if conf['cores'] == 1:  # Do not create a new process
            init_worker_fluxes(*initargs)
            for idx in range(len(params)):
                worker_fluxes(idx)
        else:  # Analyse observations in parallel
            with mp.Pool(processes=conf['cores'],
                         initializer=init_worker_fluxes,
                         initargs=initargs) as pool:
                pool.map(worker_fluxes, range(len(params)))

        return models

    def process(self, conf):
        """Process with the savedfluxes analysis.

        All the possible theoretical SED are created and the fluxes in the
        filters from the list of bands are computed and saved to a table,
        alongside the parameter values.

        Parameters
        ----------
        conf: dictionary
            Contents of pcigale.ini in the form of a dictionary
        """

        # Rename the output directory if it exists
        self.prepare_dirs()

        # Read the observations information in order to retrieve the list of
        # bands to compute the fluxes.
        observations = ObservationsManager(conf)

        # The parameters manager allows us to retrieve the models parameters
        # from a 1D index. This is useful in that we do not have to create
        # a list of parameters as they are computed on-the-fly. It also has
        # nice goodies such as finding the index of the first parameter to
        # have changed between two indices or the number of models.
        params = ParametersManager(conf)

        print("Computing the models ...")
        models = self._compute_models(conf, observations, params)

        print("Saving the models ...")
        models.save('models')


# AnalysisModule to be returned by get_module
Module = SaveFluxes
