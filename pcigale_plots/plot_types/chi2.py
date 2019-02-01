# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Copyright (C) 2013-2014 Yannick Roehlly
# Copyright (C) 2013 Institute of Astronomy
# Copyright (C) 2014 Laboratoire d'Astrophysique de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly, Médéric Boquien & Denis Burgarella

import glob
from itertools import product
from os import path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from pcigale.utils import read_table
from pcigale.analysis_modules.utils import Counter, nothread


def pool_initializer(counter):
    """Initializer of the pool of processes to share variables between workers.
    Parameters
    ----------
    :param counter: Counter class object for the number of models plotted
    """
    global gbl_counter
    # Limit the number of threads to 1 if we use MKL in order to limit the
    # oversubscription of the CPU/RAM.
    nothread()
    gbl_counter = counter


def chi2(config, outdir):
    """Plot the χ² values of analysed variables.
    """
    file = path.join(path.dirname(outdir), config.configuration['data_file'])
    input_data = read_table(file)
    chi2_vars = config.configuration['analysis_params']['variables']
    chi2_vars += [band for band in config.configuration['bands']
                  if band.endswith('_err') is False]

    items = list(product(input_data['id'], chi2_vars, [outdir]))
    counter = Counter(len(items))
    with mp.Pool(processes=config.configuration['cores'], initializer=pool_initializer,
                 initargs=(counter,)) as pool:
        pool.starmap(_chi2_worker, items)
        pool.close()
        pool.join()


def _chi2_worker(obj_name, var_name, outdir):
    """Plot the reduced χ² associated with a given analysed variable

    Parameters
    ----------
    obj_name: string
        Name of the object.
    var_name: string
        Name of the analysed variable..
    outdir: string
        The absolute path to outdir

    """
    gbl_counter.inc()
    figure = plt.figure()
    ax = figure.add_subplot(111)

    var_name = var_name.replace('/', '_')
    fnames = glob.glob("{}/{}_{}_chi2-block-*.npy".format(outdir, obj_name, var_name))
    for fname in fnames:
        data = np.memmap(fname, dtype=np.float64)
        data = np.memmap(fname, dtype=np.float64, shape=(2, data.size // 2))
        ax.scatter(data[1, :], data[0, :], color='k', s=.1)
    ax.set_xlabel(var_name)
    ax.set_ylabel("Reduced $\chi^2$")
    ax.set_ylim(0., )
    ax.minorticks_on()
    figure.suptitle("Reduced $\chi^2$ distribution of {} for {}."
                    .format(var_name, obj_name))
    figure.savefig("{}/{}_{}_chi2.pdf".format(outdir, obj_name, var_name))
    plt.close(figure)
