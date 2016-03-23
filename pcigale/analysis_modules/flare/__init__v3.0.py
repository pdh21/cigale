# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly

"""
FLARE analysis module
===========================

This module does not simulates observations withe the FLARE telescope.
It computes and save the fluxes in a set of filters and the spectra (including noise)
for all the possible combinations of input SED parameters.

"""
import ctypes
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
import time

import numpy as np

from .. import AnalysisModule
from ..utils import ParametersHandler, backup_dir, save_fluxes
from ...utils import read_table
from ...warehouse import SedWarehouse
from .workers import init_simulation as init_worker_simulation
from .workers import simulation as worker_simulation


# Limit the redshift to this number of decimals
REDSHIFT_DECIMALS = 2


class FLARE(AnalysisModule):
    """FLARE Simulation module

    This module saves figures and files corresponding to the requested model(s)
    and instrumental configuration(s) for FLARE.

    """

    parameter_list = dict([
        ("save_sfh", (
            "boolean",
            "If True, save the generated Star Formation History for each model.",
            "True"
        )),
        ('exptime', (
            'float',
            "Exposure time [sec]. Since FLARE photometric and spectroscopic observations"
            "are taken in parallel, we only need 1 exposure time",
            3600.0
        )),
        ("SNR", (
            "float",
            "What is the goal for the SNR?",
            5.0
        )),
        ("S_line", (
            "float",
            "What is the goal for S_line[erg/cm2/s]?",
            3e-18
        )),
        ("lambda_norm", (
            "float",
            "Observed wavelength[nm] of the spectrum to which the spectrum is normalised."
            "If empty, no normalisation.",
            0.
        )),
        ("mag_norm", (
            "float",
            "Magnitude used to normalise the spectrum at lambda_norm given above."
            "If empty, no normalisation.",
            0.
        )),
        ("create_table", (
            "boolean",
            "Do you want to create output tables in addition to pdf plots?",
            True
        )),
        ("flag_background", (
            "boolean",
            "If True, save the background information "
            "for each model.",
            True
        )),
        ("flag_phot", (
            "boolean",
            "If True, save the photometric sensitivity information"
            "for each model.",
            True
        )),
        ("flag_spec", (
            "boolean",
            "If True, save the spectroscopic sensitivity (continuum) information"
            "for each model.",
            True
        )),
        ("flag_line", (
            "boolean",
            "If True, save the spectroscopic sensitivity (line) information"
            "for each model.",
            True
        )),
        ("flag_sim", (
            "boolean",
            "If True, save the simulated spectroscopic observations with noises"
            "for each model.",
            True
        )),
    ])

    def process(self, data_file, column_list, creation_modules,
                creation_modules_params, parameters, cores):
        """Process with the savedfluxes analysis.

        All the possible theoretical SED are created and the fluxes in the
        filters from the column_list are computed and saved to a table,
        alongside the parameter values.

        Parameters
        ----------
        data_file: string
            Name of the file containing the observations to fit.
        column_list: list of strings
            Name of the columns from the data file to use for the analysis.
        creation_modules: list of strings
            List of the module names (in the right order) to use for creating
            the SEDs.
        creation_modules_params: list of dictionaries
            List of the parameter dictionaries for each module.
        parameters: dictionary
            Dictionary containing the parameters.
        cores: integer
            Number of cores to run the analysis on

        """

        np.seterr(invalid='ignore')

        print("Initialising the analysis module... ")

        # Rename the output directory if it exists
        backup_dir()
        save_sfh = parameters["save_sfh"].lower() == "true"
        lambda_norm = float(parameters['lambda_norm'])
        mag_norm = float(parameters['mag_norm'])
        exptime = float(parameters['exptime'])
        SNR = float(parameters['SNR'])
        S_line = float(parameters['S_line'])

        create_table = parameters["create_table"].lower() == "true"
        flag_background = parameters["flag_background"].lower() == "true"
        flag_phot = parameters["flag_phot"].lower() == "true"
        flag_spec = parameters["flag_spec"].lower() == "true"
        flag_line = parameters["flag_line"].lower() == "true"
        flag_sim = parameters["flag_sim"].lower() == "true"

        flag_phot = bool(parameters['flag_phot'])
        flag_spec = bool(parameters['flag_spec'])
        flag_line = bool(parameters['flag_line'])
        flag_sim = bool(parameters['flag_sim'])


        filters = [name for name in column_list if not name.endswith('_err')]
        n_filters = len(filters)

        if creation_modules.index('z_formation'):
            w_z_formation = creation_modules.index('z_formation')
            if list(creation_modules_params[w_z_formation]['z_formation']) == ['']:
                obs_table = read_table(data_file)
                z = np.unique(np.around(obs_table['z_formation'],
                                        decimals=REDSHIFT_DECIMALS))
                creation_modules_params[w_z_formation]['z_formation'] = z
                del obs_table, z

        # The parameters handler allows us to retrieve the models parameters
        # from a 1D index. This is useful in that we do not have to create
        # a list of parameters as they are computed on-the-fly. It also has
        # nice goodies such as finding the index of the first parameter to
        # have changed between two indices or the number of models.
        params = ParametersHandler(creation_modules, creation_modules_params)
        n_params = params.size

        # Retrieve an arbitrary SED to obtain the list of output parameters
        warehouse = SedWarehouse()
        sed = warehouse.get_sed(creation_modules, params.from_index(0))
        info = list(sed.info.keys())
        info.sort()
        n_info = len(sed.info)
        del warehouse, sed

        model_fluxes = (RawArray(ctypes.c_double,
                                 n_params * n_filters),
                        (n_params, n_filters))
        model_parameters = (RawArray(ctypes.c_double,
                                     n_params * n_info),
                            (n_params, n_info))

        print("Simulating the observations...")

        initargs = (params, filters, save_sfh, create_table, flag_background, flag_phot,
                    flag_spec, flag_line, flag_sim, lambda_norm, mag_norm, exptime, SNR,
                    S_line, model_fluxes, model_parameters, time.time(), mp.Value('i', 0))
        if cores == 1:  # Do not create a new process
            init_worker_simulation(*initargs)
            for idx in range(n_params):
                worker_simulation(idx)
        else:  # Simulate observations in parallel
            with mp.Pool(processes=cores, initializer=init_worker_simulation,
                         initargs=initargs) as pool:
                pool.map(worker_simulation, range(n_params))


# AnalysisModule to be returned by get_module
Module = FLARE
