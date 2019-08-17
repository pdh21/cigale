# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Copyright (C) 2013-2014 Yannick Roehlly
# Copyright (C) 2013 Institute of Astronomy
# Copyright (C) 2014 Laboratoire d'Astrophysique de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly, Médéric Boquien & Denis Burgarella

from astropy.table import Table
import matplotlib
import sys
from os import path

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pkg_resources
from scipy import stats
from utils.counter import Counter

# Name of the file containing the best models information
BEST_RESULTS = "results.fits"
MOCK_RESULTS = "results_mock.fits"


def pool_initializer(counter):
    """Initializer of the pool of processes to share variables between workers.
    Parameters
    ----------
    :param counter: Counter class object for the number of models plotted
    """
    global gbl_counter

    gbl_counter = counter


def mock(config, nologo, outdir):
    """Plot the comparison of input/output values of analysed variables.
    """
    best_results_file = path.abspath(path.join(outdir, BEST_RESULTS))
    mock_results_file = path.abspath(path.join(outdir, MOCK_RESULTS))

    try:
        exact = Table.read(best_results_file)
    except FileNotFoundError:
        print(f"Best models file {best_results_file} not found.")
        sys.exit(1)

    try:
        estimated = Table.read(mock_results_file)
    except FileNotFoundError:
        print(f"Mock models file {mock_results_file} not found.")
        sys.exit(1)

    params = config.configuration['analysis_params']['variables']

    for param in params:
        if param.endswith('_log'):
            param = "best."+param
            exact[param] = np.log10(exact[param[:-4]])

    logo = False if nologo else plt.imread(pkg_resources.resource_filename(__name__,
                                                                           "../resources/CIGALE.png"))

    arguments = [(exact["best."+param], estimated["bayes."+param], param, logo, outdir)
                 for param in params]

    counter = Counter(len(arguments))
    with mp.Pool(processes=config.configuration['cores'], initializer=pool_initializer,
                 initargs=(counter,)) as pool:
        pool.starmap(_mock_worker, arguments)
        pool.close()
        pool.join()


def _mock_worker(exact, estimated, param, logo, outdir):
    """Plot the exact and estimated values of a parameter for the mock analysis

    Parameters
    ----------
    exact: Table column
        Exact values of the parameter.
    estimated: Table column
        Estimated values of the parameter.
    param: string
        Name of the parameter
    nologo: boolean
        Do not add the logo when set to true.
    outdir: string
        The absolute path to outdir

    """
    gbl_counter.inc()
    range_exact = np.linspace(np.min(exact), np.max(exact), 100)

    # We compute the linear regression
    if np.min(exact) < np.max(exact):
        slope, intercept, r_value, p_value, std_err = stats.linregress(exact,
                                                                       estimated)
    else:
        slope = 0.0
        intercept = 1.0
        r_value = 0.0

    plt.errorbar(exact, estimated, marker='.', label=param, color='k',
                 linestyle='None', capsize=0.)
    plt.plot(range_exact, range_exact, color='r', label='1-to-1')
    plt.plot(range_exact, slope * range_exact + intercept, color='b',
             label=f'exact-fit $r^2$ = {r_value**2:.2f}')
    plt.xlabel('Exact')
    plt.ylabel('Estimated')
    plt.title(param)
    plt.legend(loc='best', fancybox=True, framealpha=0.5, numpoints=1)
    plt.minorticks_on()

    if logo is not False:
        plt.figimage(logo, 510, 55, origin='upper', zorder=10, alpha=1)

    plt.tight_layout()
    plt.savefig(f'{outdir}/mock_{param}.pdf')

    plt.close()
