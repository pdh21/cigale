# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Copyright (C) 2013-2014 Yannick Roehlly
# Copyright (C) 2013 Institute of Astronomy
# Copyright (C) 2014 Laboratoire d'Astrophysique de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly, Médéric Boquien & Denis Burgarella

import glob
from itertools import product
import matplotlib
from os import path

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from pcigale.utils import read_table
from pcigale.analysis_modules.utils import Counter, nothreading


def pool_initializer(counter):
    """Initializer of the pool of processes to share variables between workers.
    Parameters
    ----------
    :param counter: Counter class object for the number of models plotted
    """
    global gbl_counter
    # Limit the number of threads to 1 if we use MKL in order to limit the
    # oversubscription of the CPU/RAM.
    nothreading()
    gbl_counter = counter


def pdf(config, format, outdir):
    """Plot the PDF of analysed variables.
    """
    input_data = read_table(path.join(path.dirname(outdir), config.configuration['data_file']))
    pdf_vars = config.configuration['analysis_params']['variables']
    pdf_vars += [band for band in config.configuration['bands']
                 if band.endswith('_err') is False]

    items = list(product(input_data['id'], pdf_vars, [format], [outdir]))
    counter = Counter(len(items))
    with mp.Pool(processes=config.configuration['cores'], initializer=pool_initializer,
                 initargs=(counter,)) as pool:
        pool.starmap(_pdf_worker, items)
        pool.close()
        pool.join()


def _pdf_worker(obj_name, var_name, format, outdir):
    """Plot the PDF associated with a given analysed variable

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
    var_name = var_name.replace('/', '_')
    if var_name.endswith('_log'):
        fnames = glob.glob(f"{outdir}/{obj_name}_{var_name[:-4]}_chi2-block-"
                           f"*.npy")
        log = True
    else:
        fnames = glob.glob(f"{outdir}/{obj_name}_{var_name}_chi2-block-*.npy")
        log = False
    likelihood = []
    model_variable = []
    for fname in fnames:
        data = np.memmap(fname, dtype=np.float64)
        data = np.memmap(fname, dtype=np.float64, shape=(2, data.size // 2))

        likelihood.append(np.exp(-data[0, :] / 2.))
        model_variable.append(data[1, :])
    likelihood = np.concatenate(likelihood)
    model_variable = np.concatenate(model_variable)
    if log is True:
        model_variable = np.log10(model_variable)
    w = np.where(np.isfinite(likelihood) & np.isfinite(model_variable))
    likelihood = likelihood[w]
    model_variable = model_variable[w]

    Npdf = 100
    min_hist = np.min(model_variable)
    max_hist = np.max(model_variable)
    Nhist = min(Npdf, len(np.unique(model_variable)))

    if min_hist == max_hist:
        pdf_grid = np.array([min_hist, max_hist])
        pdf_prob = np.array([1., 1.])
    else:
        pdf_prob, pdf_grid = np.histogram(model_variable, Nhist,
                                          (min_hist, max_hist),
                                          weights=likelihood, density=True)
        pdf_x = (pdf_grid[1:]+pdf_grid[:-1]) / 2.

        pdf_grid = np.linspace(min_hist, max_hist, Npdf)
        pdf_prob = np.interp(pdf_grid, pdf_x, pdf_prob)

    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.plot(pdf_grid, pdf_prob, color='k')
    ax.set_xlabel(var_name)
    ax.set_ylabel("Probability density")
    ax.minorticks_on()
    figure.suptitle(f"Probability distribution function of {var_name} for "
                    f"{obj_name}")
    figure.savefig(f"{outdir}/{obj_name}_{var_name}_pdf.{format}")
    plt.close(figure)
