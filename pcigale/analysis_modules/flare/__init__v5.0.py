# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly

"""
FLARE analysis module
===========================

This module does not simulates observations with the FLARE telescope.
It computes and save the fluxes in a set of filters and the spectra (including noise)
for all the possible combinations of input SED parameters.

"""

from collections import OrderedDict
import ctypes
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
import time
import pyfits

import numpy as np

from .. import AnalysisModule
from ..utils import backup_dir, save_fluxes

from ...utils import read_table
from .workers import init_simulation as init_worker_simulation
from .workers import simulation as worker_simulation
from ...handlers.parameters_handler import ParametersHandler

# Directory where the output files are stored
OUT_DIR = "out/"

class FLARE(AnalysisModule):
    """FLARE Simulation module

    This module saves figures and files corresponding to the requested model(s)
    and instrumental configuration(s) for FLARE.

    """

    parameter_list = OrderedDict([
        ("variables", (
            "array of strings",
            "List of the physical properties to save. Leave empty to save all "
            "the physical properties (not recommended when there are many "
            "models).",
            None
        )),
        ("output_file", (
            "string",
            "Name of the output file that contains the modelled observations"
            "(photometry and spectra)",
            "cigale_sims"
        )),
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
            "If 0., no normalisation.",
            0.
        )),
        ("mag_norm", (
            "float",
            "Magnitude used to normalise the spectrum at lambda_norm given above."
            "If 0., no normalisation.",
            0.
        )),
        ("create_tables", (
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

    def process(self, conf):
        """Process with the savedfluxes analysis.

        All the possible theoretical SED are created and the fluxes in the
        filters from the column_list are computed and saved to a table,
        alongside the parameter values.

        Parameters
        ----------
        conf: dictionary
            Contents of pcigale.ini in the form of a dictionary
        """

        print("Initialising the analysis module... ")

        # Rename the output directory if it exists
        backup_dir()

        save_sfh = conf['analysis_method_params']['save_sfh'].lower() == "true"
        lambda_norm = float(conf['analysis_method_params']['lambda_norm'])
        mag_norm = float(conf['analysis_method_params']['mag_norm'])
        exptime = float(conf['analysis_method_params']['exptime'])
        SNR = float(conf['analysis_method_params']['SNR'])
        S_line = float(conf['analysis_method_params']['S_line'])

        create_tables = conf['analysis_method_params']['create_tables'].lower() == "true"
        flag_background = conf['analysis_method_params']['flag_background'].lower() == "true"
        flag_phot = conf['analysis_method_params']['flag_phot'].lower() == "true"
        flag_spec = conf['analysis_method_params']['flag_spec'].lower() == "true"
        flag_line = conf['analysis_method_params']['flag_line'].lower() == "true"
        flag_sim = conf['analysis_method_params']['flag_sim'].lower() == "true"

        out_file = conf['analysis_method_params']['output_file']
        #out_format = conf['analysis_method_params']['output_format']
        #save_sed = conf['analysis_method_params']['save_sed'].lower() == "true"

        # We create FLARE spectra over 2048 pixels
        n_pixels = 2048

        filters = [name for name in conf['column_list'] if not
                   name.endswith('_err')]
        n_filters = len(filters)

        # The parameters handler allows us to retrieve the models parameters
        # from a 1D index. This is useful in that we do not have to create
        # a list of parameters as they are computed on-the-fly. It also has
        # nice goodies such as finding the index of the first parameter to
        # have changed between two indices or the number of models.
        params = ParametersHandler(conf)
        n_params = params.size

        info = conf['analysis_method_params']['variables']
        n_info = len(info)

        model_spectra = (RawArray(ctypes.c_double, n_params*n_pixels),
                        (n_params, n_pixels))
        model_fluxes = (RawArray(ctypes.c_double, n_params * n_filters),
                        (n_params, n_filters))
        model_parameters = (RawArray(ctypes.c_double, n_params * n_info),
                            (n_params, n_info))

        initargs = (params, filters, info, save_sfh, create_tables, flag_background,
                    flag_phot, flag_spec, flag_line, flag_sim, lambda_norm, mag_norm,
                    exptime, SNR, S_line, model_spectra, model_fluxes,
                    model_parameters, time.time(), mp.Value('i', 0))
        if conf['cores'] == 1:  # Do not create a new process
            init_worker_simulation(*initargs)
            for idx in range(n_params):
                worker_simulation(idx)
        else:  # Create models in parallel
            with mp.Pool(processes=conf['cores'],
                         initializer=init_worker_simulation,
                         initargs=initargs) as pool:
                pool.map(worker_simulation, range(n_params))

        out_file_txt = out_file+'.txt'
        out_format_txt = 'ascii'
        save_fluxes(model_fluxes, model_parameters, filters, info, out_file_txt,
                    out_format=out_format_txt)

        out_file_fits = out_file+'.fits'
        save_spectra(model_spectra, n_params, n_pixels, model_parameters, filters,
                     info, out_file_fits)

def save_spectra(model_spectra, n_params, n_pixels, model_parameters, filters, names, out_file):
    """Save spectra fluxes and associated parameters into a table.

    Parameters
    ----------
    model_fluxes: RawArray
        Contains the fluxes of each model.
    model_parameters: RawArray
        Contains the parameters associated to each model.
    filters: list
        Contains the filter names.
    names: List
        Contains the parameters names.
    filename: str
        Name under which the file should be saved.
    directory: str
        Directory under which the file should be saved.
    out_format: str
        Format of the output file

    """
    out_spectra = np.ctypeslib.as_array(model_spectra[0])
    out_spectra = out_spectra.reshape(n_params, n_pixels)
    min_spectra = np.min(out_spectra)
    max_spectra = np.max(out_spectra)

    out_params = np.ctypeslib.as_array(model_parameters[0])
    out_params = out_params.reshape(model_parameters[1])

    out_spectra = np.rint(np.round(32768.*out_spectra/max_spectra, 0))
    #print('out_spectra 32768', out_spectra)

    hdu = pyfits.PrimaryHDU(out_spectra[:,:])
    hdu.writeto(OUT_DIR+out_file)


# AnalysisModule to be returned by get_module
Module = FLARE
