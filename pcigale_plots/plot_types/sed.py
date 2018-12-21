
import logging
from itertools import repeat
from collections import OrderedDict

from astropy.table import Table
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pkg_resources
from scipy.constants import c
from pcigale.data import Database
from pcigale.utils import read_table
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)

# Wavelength limits (restframe) when plotting the best SED.
PLOT_L_MIN = 0.1
PLOT_L_MAX = 5e5


def sed(config, sed_type, best_results_file, nologo):
    """Plot the best SED with associated observed and modelled fluxes.
    """
    obs = read_table(config.configuration['data_file'])
    mod = Table.read(best_results_file)

    with Database() as base:
        filters = OrderedDict([(name, base.get_filter(name))
                               for name in config.configuration['bands']
                               if not (name.endswith('_err') or name.startswith('line')) ])

    logo = False if nologo else plt.imread(pkg_resources.resource_filename(__name__,
                                                               "../resources/CIGALE.png"))

    with mp.Pool(processes=config.configuration['cores']) as pool:
        pool.starmap(_sed_worker, zip(obs, mod, repeat(filters),
                                      repeat(sed_type), repeat(logo)))
        pool.close()
        pool.join()


def _sed_worker(obs, mod, filters, sed_type, logo):
    """Plot the best SED with the associated fluxes in bands

    Parameters
    ----------
    obs: Table row
        Data from the input file regarding one object.
    mod: Table row
        Data from the best model of one object.
    filters: ordered dictionary of Filter objects
        The observed fluxes in each filter.
    sed_type: string
        Type of SED to plot. It can either be "mJy" (flux in mJy and observed
        frame) or "lum" (luminosity in W and rest frame)
    logo: numpy.array | boolean
        The readed logo image data. Has shape
        (M, N) for grayscale images.
        (M, N, 3) for RGB images.
        (M, N, 4) for RGBA images.
        Do not add the logo when set to False.

    """
    logger.debug("Starting worker")

    if os.path.isfile("out/{}_best_model.fits".format(obs['id'])):

        sed = Table.read("out/{}_best_model.fits".format(obs['id']))

        filters_wl = np.array([filt.pivot_wavelength
                               for filt in filters.values()])
        wavelength_spec = sed['wavelength']
        obs_fluxes = np.array([obs[filt] for filt in filters.keys()])
        obs_fluxes_err = np.array([obs[filt+'_err']
                                   for filt in filters.keys()])
        mod_fluxes = np.array([mod["best."+filt] for filt in filters.keys()])
        if obs['redshift'] >= 0:
            z = obs['redshift']
        else:  # Redshift mode
            z = mod['best.universe.redshift']
        DL = mod['best.universe.luminosity_distance']

        if sed_type == 'lum':
            xmin = PLOT_L_MIN
            xmax = PLOT_L_MAX

            k_corr_SED = 1e-29 * (4.*np.pi*DL*DL) * c / (filters_wl*1e-9)
            obs_fluxes *= k_corr_SED
            obs_fluxes_err *= k_corr_SED
            mod_fluxes *= k_corr_SED

            for cname in sed.colnames[1:]:
                sed[cname] *= wavelength_spec

            filters_wl /= 1. + z
            wavelength_spec /= 1. + z
        elif sed_type == 'mJy':
            xmin = PLOT_L_MIN * (1. + z)
            xmax = PLOT_L_MAX * (1. + z)
            k_corr_SED = 1.

            for cname in sed.colnames[1:]:
                sed[cname] *= (wavelength_spec * 1e29 /
                               (c / (wavelength_spec * 1e-9)) /
                               (4. * np.pi * DL * DL))
        else:
            logger.error("Unknown plot type")

        filters_wl /= 1000.
        wavelength_spec /= 1000.

        wsed = np.where((wavelength_spec > xmin) & (wavelength_spec < xmax))
        figure = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        if (sed.columns[1][wsed] > 0.).any():
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])

            # Stellar emission
            if 'nebular.absorption_young' in sed.columns:
                ax1.loglog(wavelength_spec[wsed],
                           (sed['stellar.young'][wsed] +
                            sed['attenuation.stellar.young'][wsed] +
                            sed['nebular.absorption_young'][wsed] +
                            sed['stellar.old'][wsed] +
                            sed['attenuation.stellar.old'][wsed] +
                            sed['nebular.absorption_old'][wsed]),
                           label="Stellar attenuated ", color='orange',
                           marker=None, nonposy='clip', linestyle='-',
                           linewidth=0.5)
            else:
                ax1.loglog(wavelength_spec[wsed],
                           (sed['stellar.young'][wsed] +
                            sed['attenuation.stellar.young'][wsed] +
                            sed['stellar.old'][wsed] +
                            sed['attenuation.stellar.old'][wsed]),
                           label="Stellar attenuated ", color='orange',
                           marker=None, nonposy='clip', linestyle='-',
                           linewidth=0.5)
            ax1.loglog(wavelength_spec[wsed],
                       (sed['stellar.old'][wsed] +
                        sed['stellar.young'][wsed]),
                       label="Stellar unattenuated", color='b', marker=None,
                       nonposy='clip', linestyle='--', linewidth=0.5)
            # Nebular emission
            if 'nebular.lines_young' in sed.columns:
                ax1.loglog(wavelength_spec[wsed],
                           (sed['nebular.lines_young'][wsed] +
                            sed['nebular.lines_old'][wsed] +
                            sed['nebular.continuum_young'][wsed] +
                            sed['nebular.continuum_old'][wsed] +
                            sed['attenuation.nebular.lines_young'][wsed] +
                            sed['attenuation.nebular.lines_old'][wsed] +
                            sed['attenuation.nebular.continuum_young'][wsed] +
                            sed['attenuation.nebular.continuum_old'][wsed]),
                           label="Nebular emission", color='y', marker=None,
                           nonposy='clip', linewidth=.5)
            # Dust emission Draine & Li
            if 'dust.Umin_Umin' in sed.columns:
                ax1.loglog(wavelength_spec[wsed],
                           (sed['dust.Umin_Umin'][wsed] +
                            sed['dust.Umin_Umax'][wsed]),
                           label="Dust emission", color='r', marker=None,
                           nonposy='clip', linestyle='-', linewidth=0.5)
            # Dust emission Dale
            if 'dust' in sed.columns:
                ax1.loglog(wavelength_spec[wsed], sed['dust'][wsed],
                           label="Dust emission", color='r', marker=None,
                           nonposy='clip', linestyle='-', linewidth=0.5)
            # AGN emission Fritz
            if 'agn.fritz2006_therm' in sed.columns:
                ax1.loglog(wavelength_spec[wsed],
                           (sed['agn.fritz2006_therm'][wsed] +
                            sed['agn.fritz2006_scatt'][wsed] +
                            sed['agn.fritz2006_agn'][wsed]),
                           label="AGN emission", color='g', marker=None,
                           nonposy='clip', linestyle='-', linewidth=0.5)
            # Radio emission
            if 'radio_nonthermal' in sed.columns:
                ax1.loglog(wavelength_spec[wsed],
                           sed['radio_nonthermal'][wsed],
                           label="Radio nonthermal", color='brown',
                           marker=None, nonposy='clip', linestyle='-',
                           linewidth=0.5)

            ax1.loglog(wavelength_spec[wsed], sed['L_lambda_total'][wsed],
                       label="Model spectrum", color='k', nonposy='clip',
                       linestyle='-', linewidth=1.5)

            ax1.set_autoscale_on(False)
            s = np.argsort(filters_wl)
            filters_wl = filters_wl[s]
            mod_fluxes = mod_fluxes[s]
            obs_fluxes = obs_fluxes[s]
            obs_fluxes_err = obs_fluxes_err[s]
            ax1.scatter(filters_wl, mod_fluxes, marker='o', color='r', s=8,
                        zorder=3, label="Model fluxes")
            mask_ok = np.logical_and(obs_fluxes > 0., obs_fluxes_err > 0.)
            ax1.errorbar(filters_wl[mask_ok], obs_fluxes[mask_ok],
                         yerr=obs_fluxes_err[mask_ok]*3, ls='', marker='s',
                         label='Observed fluxes', markerfacecolor='None',
                         markersize=6, markeredgecolor='b', capsize=0.)
            mask_uplim = np.logical_and(np.logical_and(obs_fluxes > 0.,
                                                       obs_fluxes_err < 0.),
                                        obs_fluxes_err > -9990. * k_corr_SED)
            if not mask_uplim.any() == False:
                ax1.errorbar(filters_wl[mask_uplim], obs_fluxes[mask_uplim],
                             yerr=obs_fluxes_err[mask_uplim]*3, ls='',
                             marker='v', label='Observed upper limits',
                             markerfacecolor='None', markersize=6,
                             markeredgecolor='g', capsize=0.)
            mask_noerr = np.logical_and(obs_fluxes > 0.,
                                        obs_fluxes_err < -9990. * k_corr_SED)
            if not mask_noerr.any() == False:
                ax1.errorbar(filters_wl[mask_noerr], obs_fluxes[mask_noerr],
                             ls='', marker='s', markerfacecolor='None',
                             markersize=6, markeredgecolor='r',
                             label='Observed fluxes, no errors', capsize=0.)
            mask = np.where(obs_fluxes > 0.)
            ax2.errorbar(filters_wl[mask],
                         (obs_fluxes[mask]-mod_fluxes[mask])/obs_fluxes[mask],
                         yerr=obs_fluxes_err[mask]/obs_fluxes[mask]*3,
                         marker='_', label="(Obs-Mod)/Obs", color='k',
                         capsize=0.)
            ax2.plot([xmin, xmax], [0., 0.], ls='--', color='k')
            ax2.set_xscale('log')
            ax2.minorticks_on()

            figure.subplots_adjust(hspace=0., wspace=0.)

            ax1.set_xlim(xmin, xmax)
            ymin = min(np.min(obs_fluxes[mask_ok]),
                       np.min(mod_fluxes[mask_ok]))
            if not mask_uplim.any() == False:
                ymax = max(max(np.max(obs_fluxes[mask_ok]),
                               np.max(obs_fluxes[mask_uplim])),
                           max(np.max(mod_fluxes[mask_ok]),
                               np.max(mod_fluxes[mask_uplim])))
            else:
                ymax = max(np.max(obs_fluxes[mask_ok]),
                           np.max(mod_fluxes[mask_ok]))
            ax1.set_ylim(1e-1*ymin, 1e1*ymax)
            ax2.set_xlim(xmin, xmax)
            ax2.set_ylim(-1.0, 1.0)
            if sed_type == 'lum':
                ax2.set_xlabel("Rest-frame wavelength [$\mu$m]")
                ax1.set_ylabel("Luminosity [W]")
                ax2.set_ylabel("Relative residual luminosity")
            else:
                ax2.set_xlabel("Observed wavelength [$\mu$m]")
                ax1.set_ylabel("Flux [mJy]")
                ax2.set_ylabel("Relative residual flux")
            ax1.legend(fontsize=6, loc='best', fancybox=True, framealpha=0.5)
            ax2.legend(fontsize=6, loc='best', fancybox=True, framealpha=0.5)
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax1.get_yticklabels()[1], visible=False)
            figure.suptitle("Best model for {} at z = {}. Reduced $\chi^2$={}".
                            format(obs['id'], np.round(z, decimals=3),
                                   np.round(mod['best.reduced_chi_square'],
                                            decimals=2)))
            if logo is not False:
                figure_height = figure.get_figheight() * figure.dpi
                figure.figimage(logo, 12, figure_height - 67, origin='upper', zorder=0,
                                alpha=1)

            figure.savefig("out/{}_best_model.pdf".format(obs['id']))
            plt.close(figure)
        else:
            logger.error("No valid best SED found for {}. No plot created.".format(obs['id']))
    else:
        logger.error("No SED found for {}. No plot created.".format(obs['id']))
