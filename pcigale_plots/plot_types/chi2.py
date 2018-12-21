
import glob
from itertools import product

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from pcigale.utils import read_table


def chi2(config):
    """Plot the χ² values of analysed variables.
    """
    input_data = read_table(config.configuration['data_file'])
    chi2_vars = config.configuration['analysis_params']['variables']
    chi2_vars += [band for band in config.configuration['bands']
                  if band.endswith('_err') is False]

    with mp.Pool(processes=config.configuration['cores']) as pool:
        items = product(input_data['id'], chi2_vars)
        pool.starmap(_chi2_worker, items)
        pool.close()
        pool.join()


def _chi2_worker(obj_name, var_name):
    """Plot the reduced χ² associated with a given analysed variable

    Parameters
    ----------
    obj_name: string
        Name of the object.
    var_name: string
        Name of the analysed variable..

    """
    figure = plt.figure()
    ax = figure.add_subplot(111)

    var_name = var_name.replace('/', '_')
    fnames = glob.glob("out/{}_{}_chi2-block-*.npy".format(obj_name, var_name))
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
    figure.savefig("out/{}_{}_chi2.pdf".format(obj_name, var_name))
    plt.close(figure)
