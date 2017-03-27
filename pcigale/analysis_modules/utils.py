# -*- coding: utf-8 -*-
# Copyright (C) 2014 Médéric Boquien
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Médéric Boquien

"""
Various utility functions for pcigale analysis modules
"""

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
