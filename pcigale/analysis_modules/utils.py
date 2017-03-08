# -*- coding: utf-8 -*-
# Copyright (C) 2014 Médéric Boquien
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Médéric Boquien

"""
Various utility functions for pcigale analysis modules
"""

import numpy as np
from astropy import log
from astropy.table import Table, Column

log.setLevel('ERROR')

def save_fluxes(model_fluxes, model_parameters, filters, names):
    """Save fluxes and associated parameters into a table.

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

    """
    out_fluxes = np.ctypeslib.as_array(model_fluxes[0])
    out_fluxes = out_fluxes.reshape(model_fluxes[1])

    out_params = np.ctypeslib.as_array(model_parameters[0])
    out_params = out_params.reshape(model_parameters[1])

    out_table = Table(np.hstack((out_fluxes, out_params)),
                      names=filters + list(names))

    out_table.add_column(Column(np.arange(model_fluxes[1][0]), name='id'),
                         index=0)

    out_table.write("out/computed_fluxes.fits")
    out_table.write("out/computed_fluxes.txt", format='ascii.fixed_width',
                    delimiter=None)


def nothread():
    """Some libraries such as Intel's MKL have automatic threading. This is
    good when having only one process. However we already do our own
    parallelisation. The additional threads created by the MKL increase in
    excess the pressure on the CPU and on the RAM slowing everything down.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    try:
        import mkl
        mkl.set_num_threads(1)
    except ImportError:
        pass
