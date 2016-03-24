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
import random
from astropy.cosmology import WMAP5 # WMAP 5-year cosmology
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import tee, islice, chain
import numpy as np

from .. import AnalysisModule
from ..utils import backup_dir, save_fluxes

from ...utils import read_table
from .workers import init_simulation as init_worker_simulation
from .workers import simulation as worker_simulation
from ...handlers.parameters_handler import ParametersHandler

# Directory where the output files are stored
OUT_DIR = "out/"

def previous_and_next(some_iterable):
    prevs, items, nexts = tee(some_iterable, 3)
    prevs = chain([None], prevs)
    nexts = chain(islice(nexts, 1, None), [None])
    return zip(prevs, items, nexts)

def plot_N_z(x, y, sample):

    fig, ax1 = plt.subplots(1, 1)

    ax1.plot(x, y)
    ax1.hist(np.sort(sample), len(x))

    ymin = 0.1*max(np.min(y), 1./3600.)
    ymax = 5.0*np.max(y)
    ax1.set_ylim(ymin, ymax)
    ax1.set_yscale('log')

    ax1.set_xlabel('redshift', fontsize=20)
    ax1.set_ylabel('N$_{galaxies}$ $arcmin^{-2}$ $(dz=1)^{-1}$', fontsize=20)
    ax1.legend(loc=0, fontsize=8)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylim(np.log10(3600.*ymin), np.log10(3600.*ymax))
    ax2.set_ylabel('$log_{10}$ N$_{galaxies}$ $deg^{-2}$ $(dz=1)^{-1}$', fontsize=20)

    plt.show()

def montecarlo(x, y):

    from scipy.stats import rv_discrete

    # Safety check on arrays sizes
    if len(x) != len(y):
        sys.exit('x and y arrays must have the same size.')

    sum_y = np.sum(y)
    y /= sum_y

    # Prepare array for random sample
    distribution = rv_discrete(values=(range(len(x)), y))
    sample = x[distribution.rvs(size=sum_y)]
    #print('Size of sample: ', len(sample))
    y *= sum_y

    plot_N_z(x, y, sample)

    return sample

def func(x, a, b, c, d):
    from scipy.special import erf, erfc
    #return a*np.log10(x/b)+c

    return a*(np.arctan(x**b/c)) - d
    #return x**a*(np.arctan((x-b)/c)) - d

    #return a*(x**c)

def density_m(FoV_axis1, FoV_axis2, redshifts):
    """
    Parameters
    ----------

    """
    from scipy.optimize import curve_fit
    from astropy.cosmology import WMAP7 as cosmology
    from scipy.integrate import simps

    # Conversion log10[M_Salpeter (1955)] = log10[M_Chabrier (2003)] + 0.24:
    # http://www.brera.inaf.it/utenti/marcella/masse3.html

    # Model: Furlong et al. (2015) with Chabrier (2003)
    logMstar_1  = np.array([11.14, 11.11, 11.06, 10.91, 10.78, 10.60]) + 0.24
    Phi_star_1  = np.array([ 0.84,  0.84,  0.74,  0.45,  0.22,  0.12])*1e-3
    alpha_1     = np.array([-1.43, -1.45, -1.48, -1.57, -1.66, -1.74])
    z_1         = np.array([  0.1,   0.5,   1.0,   2.0,   3.0,   4.0])
    # Data: Song et al. (2015) with Salpeter (1955)
    logMstar_2  = np.array([10.44, 10.47, 10.30, 10.42, 10.41])
    Phi_star_2  = np.array([30.13, 13.36,  3.03,  0.70,  0.03])*1e-5/1.65
    alpha_2     = np.array([-1.53, -1.67, -1.93, -2.05, -2.45])
    z_2         = np.array([ 4.0,    5.0,   6.0,   7.0,   8.0])
    #logMstar_2  = np.array([10.44, 10.47, 10.30, 10.42])
    #Phi_star_2  = np.array([30.13, 13.36,  3.03,  0.70])*1e-5
    #alpha_2     = np.array([-1.53, -1.67, -1.93, -2.05])
    #z_2         = np.array([ 4.0,    5.0,   6.0,   7.0])
    # Data: Duncan et al. (2014) with Chabrier (2003)
    logMstar_3  = np.array([10.51, 10.68, 10.87, 10.51]) + 0.24
    Phi_star_3  = np.array([ 1.89,  1.24,  0.14,  0.36])*1e-4
    alpha_3     = np.array([-1.89, -1.74, -2.00, -1.89])
    z_3         = np.array([  4.0,   5.0,   6.0,   7.0])
    # Data: Grazian et al. (2015) with Salpeter (1955)
    logMstar_4    = np.array([10.96, 10.78, 10.49, 10.69])
    logPhi_star_4 = np.array([-3.94, -4.18, -4.16, -5.24])
    Phi_star_4    = 10**logPhi_star_4/1.65
    alpha_4       = np.array([-1.63, -1.63, -1.55, -1.88])
    zmin_4        = np.array([  3.5,   4.5,   5.5,   6.5])
    zmax_4        = np.array([  4.5,   5.5,   6.5,   7.5])
    z_4           = (zmin_4 + zmax_4)/2.
    # Data: Mortlock et al. (2014) with Chabrier (2003)
    logMstar_5    = np.array([10.90, 10.90, 11.04, 11.15, 11.02, 11.04]) + 0.24
    logPhi_star_5 = np.array([-2.54, -2.71, -3.21, -3.74, -3.78, -4.03])
    Phi_star_5    = 10**logPhi_star_5
    alpha_5       = np.array([-1.59, -1.42, -1.31, -1.51, -1.56, -1.69])
    zmin_5        = np.array([  0.3,   0.5,   1.0,   1.5,   2.0,   2.5])
    zmax_5        = np.array([  0.5,   1.0,   1.5,   2.0,   2.5,   3.0])
    z_5           = (zmin_5 + zmax_5)/2.
    # Data: Caputi et al. (2015) with Salpeter (1955)
    logMstar_6    = np.array([11.26, 11.38])
    Phi_star_6    = np.array([ 1.62,  4.39])*1e-4
    alpha_6       = np.array([-1.72, -1.88])
    zmin_6        = np.array([  3.0,   4.0])
    zmax_6        = np.array([  4.0,   5.0])
    z_6           = (zmin_6 + zmax_6)/2.

    z_data = np.hstack((z_1, z_2, z_3, z_4, z_5, z_6))
    x_data = cosmology.age(z_data).value * 1000.
    N_ages = 100
    ages_data = np.linspace(1., cosmology.age(0.).value * 1000., N_ages)

    #Phi_M = (Phi_star*/M_star) * (M/M_star)**alpha * np.exp(-M/M_star)
    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(cosmology.age(z_1).value * 1000., alpha_1, 'r', marker='+', linestyle='None', markersize=12, label=" Furlong et al. (2015)")
    ax1.plot(cosmology.age(z_2).value * 1000., alpha_2, 'g', marker='x', linestyle='None', markersize=12, label=" Song et al. (2015).")
    ax1.plot(cosmology.age(z_3).value * 1000., alpha_3, 'b', marker='s', linestyle='None', markersize=12, label=" Duncan et al. (2013)")
    ax1.plot(cosmology.age(z_4).value * 1000., alpha_4, 'r', marker='o', linestyle='None', markersize=12, label=" Grazian et al. (2015)")
    ax1.plot(cosmology.age(z_5).value * 1000., alpha_5, 'g', marker='.', linestyle='None', markersize=12, label=" Mortlock et al. (2014)")
    ax1.plot(cosmology.age(z_6).value * 1000., alpha_6, 'b', marker='*', linestyle='None', markersize=12, label=" Caputi et al. (2015)")
    ax1.set_xlabel('age [Myr]', fontsize=20)
    ax1.set_ylabel('alpha', fontsize=20)
    ax1.legend(loc=0, fontsize=8)
    #ax1.set_xscale('log')
    #ax1.set_yscale('log')
    y_data = np.hstack((alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6))
    popt1, pcov1 = curve_fit(func, x_data, y_data, p0=(2.5, 1.5, 8000., 10.),
                       bounds=([1.0, 1., 5000., 2.0], [100.0, 5., 10000., 10.]))
    #popt1, pcov1 = curve_fit(func, x_data, y_data, p0=(0.08, 450., 50., 4.),)
    #                   bounds=([0.00, 400., 10., 2.0], [0.10, 800., 60., 6.]))

    #print("a =", popt1[0], "+/-", pcov1[0,0]**0.5)
    #print("b =", popt1[1], "+/-", pcov1[1,1]**0.5)
    #print("c =", popt1[2], "+/-", pcov1[2,2]**0.5)
    #print("d =", popt1[3], "+/-", pcov1[2,2]**0.5)
    ax1.plot(ages_data, func(ages_data, *popt1), 'r-', lw=3)
    ax1.set_xlim(1., cosmology.age(0.).value * 1000.)

    fig, ax2 = plt.subplots(1, 1)
    ax2.plot(cosmology.age(z_1).value * 1000., logMstar_1, 'r', marker='+', linestyle='None', markersize=12, label=" Furlong et al. (2015)")
    ax2.plot(cosmology.age(z_2).value * 1000., logMstar_2, 'g', marker='x', linestyle='None', markersize=12, label=" Song et al. (2015).")
    ax2.plot(cosmology.age(z_3).value * 1000., logMstar_3, 'b', marker='s', linestyle='None', markersize=12, label=" Duncan et al. (2013)")
    ax2.plot(cosmology.age(z_4).value * 1000., logMstar_4, 'r', marker='o', linestyle='None', markersize=12, label=" Grazian et al. (2015)")
    ax2.plot(cosmology.age(z_5).value * 1000., logMstar_5, 'g', marker='.', linestyle='None', markersize=12, label=" Mortlock et al. (2014)")
    ax2.plot(cosmology.age(z_6).value * 1000., logMstar_6, 'b', marker='*', linestyle='None', markersize=12, label=" Caputi et al. (2015)")
    ax2.set_xlabel('age [Myr]', fontsize=20)
    ax2.set_ylabel('log10 M${\star}$', fontsize=20)
    ax2.legend(loc=0, fontsize=8)
    #ax2.set_xscale('log')
    #ax2.set_yscale('log')
    y_data = np.hstack((logMstar_1, logMstar_2, logMstar_3, logMstar_4, logMstar_5, logMstar_6))
    popt2, pcov2 = curve_fit(func, x_data, y_data, p0=(2.5, 1.4, 1000., -8.0),
                       bounds=([2.4, 1.2, 1000., -8.0], [2.6, 1.6, 2000., -7.0]))

    #popt2, pcov2 = curve_fit(func, x_data, y_data, p0=(0.10, 30., 500, -9.),
    #                   bounds=([0.00, 0., 50., -11.0], [0.15, 50., 1000., -7.]))
    #print("a =", popt2[0], "+/-", pcov2[0,0]**0.5)
    #print("b =", popt2[1], "+/-", pcov2[1,1]**0.5)
    #print("c =", popt2[2], "+/-", pcov2[2,2]**0.5)
    #print("d =", popt2[3], "+/-", pcov2[2,2]**0.5)
    ax2.plot(ages_data, func(ages_data, *popt2), 'r-', lw=3)
    ax2.set_xlim(1., cosmology.age(0.).value * 1000.)

    fig, ax3 = plt.subplots(1, 1)
    ax3.plot(cosmology.age(z_1).value * 1000., np.log10(Phi_star_1), 'r', marker='+', linestyle='None', markersize=12, label=" Furlong et al. (2015)")
    ax3.plot(cosmology.age(z_2).value * 1000., np.log10(Phi_star_2), 'g', marker='x', linestyle='None', markersize=12, label=" Song et al. (2015).")
    ax3.plot(cosmology.age(z_3).value * 1000., np.log10(Phi_star_3), 'b', marker='s', linestyle='None', markersize=12, label=" Duncan et al. (2013)")
    ax3.plot(cosmology.age(z_4).value * 1000., np.log10(Phi_star_4), 'r', marker='o', linestyle='None', markersize=12, label=" Grazian et al. (2015)")
    ax3.plot(cosmology.age(z_5).value * 1000., np.log10(Phi_star_5), 'g', marker='.', linestyle='None', markersize=12, label=" Mortlock et al. (2014)")
    ax3.plot(cosmology.age(z_6).value * 1000., np.log10(Phi_star_6), 'b', marker='*', linestyle='None', markersize=12, label=" Caputi et al. (2016)")
    ax3.set_xlabel('age [Myr]', fontsize=20)
    ax3.set_ylabel('log10 Phi${\star}$', fontsize=20)
    ax3.legend(loc=0, fontsize=8)
    #ax3.set_xscale('log')
    #ax3.set_yscale('log')
    y_data = np.hstack((Phi_star_1, Phi_star_2, Phi_star_3, Phi_star_4, Phi_star_5, Phi_star_6))
    logy_data = np.log10(y_data)

    popt3, pcov3 = curve_fit(func, x_data, logy_data, p0=(1., 3., 15000., 15.0),
                       bounds=([1.0, 1., 10000., 10.0], [100.0, 5., 20000., 20.]))

    #popt3, pcov3 = curve_fit(func, x_data, logy_data, p0=(0.12, 550., 129, 6.),
    #                   bounds=([0.05, 400., 10., 5.0], [0.15, 900., 200., 9.]))
    #print("a =", popt3[0], "+/-", pcov3[0,0]**0.5)
    #print("b =", popt3[1], "+/-", pcov3[1,1]**0.5)
    #print("c =", popt3[2], "+/-", pcov3[2,2]**0.5)
    #print("d =", popt3[3], "+/-", pcov3[2,2]**0.5)
    ax3.plot(ages_data, func(ages_data, *popt3), 'r-', lw=3)
    ax3.set_xlim(1., cosmology.age(0.).value * 1000.)
    #plt.show()

    # We convert from /(100 Mpc/h)**3 to /kpc**3, i.e.,
    # (100 Mpc)**3 = (100 * 1000)**3 = 1e15 and to /arcmin2/dz
    sum = 0.
    fig, ax4 = plt.subplots(1, 1)
    cmap = mpl.cm.rainbow
    SMD = []
    for previous, item, next in previous_and_next(redshifts):
        if item == 0:
            previous = 0
        if previous is None:
            previous = item - min(next - item, item)
        if next is None:
            next = item + (item - previous)
        l = FoV_axis1*(1e-3*cosmology.kpc_proper_per_arcmin(next)).value
        h = FoV_axis2*(1e-3*cosmology.kpc_proper_per_arcmin(next)).value
        L = (cosmology.comoving_distance(next).value-
                cosmology.comoving_distance(item).value)
        volume = L * l * h
        #print('z, l, h, L volume', item, FoV_axis1, FoV_axis2, l, h, L, volume)
        z = (item + next)/2.

        # We compute alpha, log10(M_star) and log10(Phi_star)
        alpha = func(cosmology.age(z).value * 1000., *popt1)
        logM_star = (func(cosmology.age(z).value * 1000., *popt2))
        logPhi_star = (func(cosmology.age(z).value * 1000., *popt3))
        #print('LF', item, cosmology.age(z).value * 1000., alpha, np.log10(M_star), np.log10(Phi_star))

        N_masses = 20
        # We sample the stellar masses range, logarithmically
        logM = np.linspace(8, 13, N_masses)
        Phi_M = np.log(10)*10**logPhi_star * 10**((logM-logM_star)*(1+alpha)) * np.exp(-10**(logM-logM_star))
        #Phi_M = (Phi_star/M_star) * (M/M_star)**alpha * np.exp(-(M/M_star))

        I = simps(Phi_M, 10**logM)
        I2 = np.trapz(Phi_M, 10**logM)
        SMD.append(np.log10(I2))
        sum +=  I*volume
        #print('SMD', item, volume, I2, np.log10(I2), I2*volume, sum)
        #print('SMD', item, volume, I2, 8+np.log10(I2), I*volume, sum)
        ax4.plot(logM, Phi_M, color=cmap(item/max(redshifts)))
        #ax4.set_xscale('log')
        ax4.set_yscale('log')
    ax4.set_xlabel('$log_{10}$ M${\star}$', fontsize=20)
    ax4.set_ylabel('$log_{10}$ Phi$_\star$ dex$^{-1}$', fontsize=20)
    ax4.text(0.6, 0.95, 'CIGALE Mass Functions from z = %.1f to z = %.1f' %(np.min(redshifts), np.max(redshifts)), ha='center', va='center', transform=ax4.transAxes)
    ax4.legend(loc=0, fontsize=6)
    ax4.grid(True)
    ax4.set_xlim(8, 13)
    ax4.set_ylim(1e-10, 1e-1)

    fig, ax5 = plt.subplots(1, 1)
    ax5.plot(redshifts, SMD)
    ax5.set_xlabel('redshift', fontsize=20)
    ax5.set_ylabel('$log_{10}$ N [Mpc$^{-3}$ M$_\odot$]', fontsize=20)
    #ax5.text(0.6, 0.95, 'CIGALE Mass Functions from z = %.1f to z = %.1f' %(np.min(redshifts), np.max(redshifts)), ha='center', va='center', transform=ax4.transAxes)
    ax5.legend(loc=0, fontsize=6)
    ax5.grid(True)
    #ax5.set_xlim(1, 20)
    #ax5.set_ylim(-8, 8)

    plt.show()
    #m_sample = montecarlo(m, N_m.value)

    # Random create of RA and Dec within the FoV
    #RA_sample = [60.*random.uniform(0., FoV_axis1)
    #             for ind in range(len(z_sample))]
    #Dec_sample = [60.*random.uniform(0, FoV_axis2)
     #            for ind in range(len(z_sample))]

    #return RA_sample, Dec_sample, m_sample, z_sample



def density_z(FoV_axis1, FoV_axis2, z):
    """
    Parameters
    ----------
    How many objects per redshift bin and per solid angle (FoV_axis1 x FoV_axis1 arcmin2)?
    Initial data: side = 100 Mpc/h comobile with h = h0*(1+z), avec h0 = 0.702 WMAP-5yr

    """
    h0 = 0.702

    # We start by computing the redshift bins
    delta_z = []
    dz = []
    #ind_z = 0
    #z_morgane = [3.989, 4.334, 4.746, 5.247, 5.873, 6.676, 7.570, 8.569, 9.684]
    for previous, item, next in previous_and_next(z):
        if item == 0:
            previous = 0
        if previous is None:
            previous = item - min(next - item, item)
        if next is None:
            next = item + (item - previous)
        delta_z.append((next - previous) / 2.)
        dz.append(1e3*(WMAP5.comoving_distance(item+delta_z[-1]/2.).value-
                  WMAP5.comoving_distance(item-delta_z[-1]/2.).value))
        #print('z_min, z, z_max, delta_z, dz', item-delta_z[ind_z]/2., z[ind_z],
        #      item+delta_z[ind_z]/2., delta_z[ind_z], dz[ind_z])
        #ind_z += 1

    # We compute the distribution of galaxies as a function of the redshift
    # using a "fit" on Morgane Cousin's model in comobile (100 Mpc/h)**3
    N_z = [7.e6*(z/1.5)*np.exp(-z/1.5) for z in z]
    #Sum_N_z = 0.
    #for ind_z in range(len(z)):
    #    Sum_N_z += 1e-3*N_z[ind_z]*dz[ind_z]
    #print('Sum N_z', Sum_N_z)

    #N_z_2 = [2.e3*(z/1.00)*np.exp(-z/1.00) for z in z]

    # We convert from /(100 Mpc/h)**3 to /kpc**3, i.e.,
    # (100 Mpc)**3 = (100 * 1000)**3 = 1e15 and to /arcmin2/dz
    N_z *= 1e-15 * h0**3 *(WMAP5.kpc_proper_per_arcmin(z))**2 \
              * FoV_axis1 * FoV_axis2 * dz / delta_z
    #Sum_N_z = 0.
    #for ind_z in range(len(z)):
    #    print('z = ', z[ind_z], 'kpc_proper_per_arcmin(z) = ', N_z[ind_z])
    #    Sum_N_z += N_z[ind_z]*delta_z[ind_z]
    #print('Sum N_z', Sum_N_z)

    z_sample = montecarlo(z, N_z.value)

    # Random create of RA and Dec within the FoV
    RA_sample = [60.*random.uniform(0., FoV_axis1)
                 for ind in range(len(z_sample))]
    Dec_sample = [60.*random.uniform(0, FoV_axis2)
                 for ind in range(len(z_sample))]

    return RA_sample, Dec_sample, z_sample

class FLARE(AnalysisModule):
    """FLARE Simulation module

    This module saves figures and files corresponding to the requested model(s)
    and instrumental configuration(s) for FLARE.

    """

    parameter_list = OrderedDict([
        ("variables", (
            "cigale_list(minvalue=0., maxvalue=30.)",
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
        ("FoV_axis1", (
            "float",
            "Field ov View (arcmin)",
            1.0
        )),
        ("FoV_axis2", (
            "float",
            "Field ov View (arcmin)",
            1.0
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

        save_sfh = conf['analysis_params']['save_sfh']
        lambda_norm = float(conf['analysis_params']['lambda_norm'])
        mag_norm = float(conf['analysis_params']['mag_norm'])
        exptime = float(conf['analysis_params']['exptime'])
        SNR = float(conf['analysis_params']['SNR'])
        S_line = float(conf['analysis_params']['S_line'])

        create_tables = conf['analysis_params']['create_tables']
        flag_background = conf['analysis_params']['flag_background']
        flag_phot = conf['analysis_params']['flag_phot']
        flag_spec = conf['analysis_params']['flag_spec']
        flag_line = conf['analysis_params']['flag_line']
        flag_sim = conf['analysis_params']['flag_sim']

        out_file = conf['analysis_params']['output_file']

        # FOV in arcmin
        FoV_axis1 = float(conf['analysis_params']['FoV_axis1'])
        FoV_axis2 = float(conf['analysis_params']['FoV_axis2'])

        redshifts = conf['sed_modules_params']['z_formation']['redshift']
        #RA_sample, Dec_sample, z_sample = density_z(FoV_axis1, FoV_axis2, redshifts)
        #print('Echantillon', len(RA_sample), len(Dec_sample), len(z_sample))
        RA_sample, Dec_sample, z_sample, M_sample = density_m(
                                            FoV_axis1, FoV_axis2, redshifts)

        # We create FLARE spectra over 2048 pixels with 0.4x0.366-arcsec2 pixels
        slice_length = 25 # arcsec
        slice_width = 0.4 # arcsec
        pixel_spec1 = 0.366 # arcsec/pixel for spectroscopy
        pixel_spec2 = slice_width # arcsec/pixel for spectroscopy
        naxis1 = int(round(FoV_axis1 * FoV_axis2 * 3600. / (pixel_spec1 * pixel_spec2), 0))
        n_pixel2 = round(slice_length / pixel_spec2, 0)
        n_pixels = 2048
        naxis2 = n_pixels

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

        info = conf['analysis_params']['variables']
        n_info = len(info)

        model_spectra = (RawArray(ctypes.c_double, n_params*n_pixels),
                        (n_params, n_pixels))
        model_background = (RawArray(ctypes.c_double, naxis1*naxis2),
                        (naxis1, naxis2))
        model_redshift = (RawArray(ctypes.c_double, n_params),
                        (n_params))
        model_fluxes = (RawArray(ctypes.c_double, n_params * n_filters),
                        (n_params, n_filters))
        model_parameters = (RawArray(ctypes.c_double, n_params * n_info),
                            (n_params, n_info))

        initargs = (params, filters, info, save_sfh, create_tables, flag_background,
                    flag_phot, flag_spec, flag_line, flag_sim, lambda_norm, mag_norm,
                    exptime, SNR, S_line, model_spectra, model_background,
                    model_redshift, model_fluxes,
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
        save_spectra(RA_sample, Dec_sample, z_sample, model_spectra, model_background,
                     model_redshift, naxis1, naxis2,
                     n_params, n_pixels, model_parameters, filters, info, out_file_fits,
                     FoV_axis1, FoV_axis2, pixel_spec1, pixel_spec2)

def save_spectra(RA_sample, Dec_sample, z_sample, model_spectra, model_background,
                 model_redshift, naxis1, naxis2,
                 n_params, n_pixels, model_parameters, filters, names, out_file,
                 FoV_axis1, FoV_axis2, pixel_spec1, pixel_spec2):
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
    from random import randint

    # array containing the redshift of each model
    out_redshift = np.ctypeslib.as_array(model_redshift[0])
    out_redshift = out_redshift.reshape(n_params)

    # array containing each modelled spectrum
    out_background = np.ctypeslib.as_array(model_background[0])
    out_background = out_background.reshape(naxis1, naxis2)

    out_spectra = np.ctypeslib.as_array(model_spectra[0])
    out_spectra = out_spectra.reshape(n_params, n_pixels)
    min_spectra = np.min(out_spectra)
    max_spectra = np.max(out_spectra)

    # array containing the parameters, if any, associated to each model
    out_params = np.ctypeslib.as_array(model_parameters[0])
    out_params = out_params.reshape(model_parameters[1])

    # We convert the spectra to integers [0 - 32768]
    out_spectra = out_spectra/max_spectra

    hdu = pyfits.PrimaryHDU(out_spectra[:,:])
    hdu.writeto(OUT_DIR+out_file)

    filename=OUT_DIR+'simobs.dat'
    param_names =  ' '.join(names)
    simulation = open(filename, 'w')
    simulation.write('%s %s %s' %('# Row model# redshift RA Dec ', param_names, '\n'))
    #simulation.write('# Row model# redshift RA Dec param_names \n')

    naxis1 = int(round(FoV_axis1 * FoV_axis2 * 3600. / (pixel_spec1 * pixel_spec2), 0))
    slice_length = 25 # arcsec
    n_pixel2 = round(slice_length / pixel_spec2, 0)
    naxis2 = n_pixels
    spectral_image = np.zeros((naxis1, naxis2))

    for i in range(naxis1):
        l = randint(0, n_params-1)
        spectral_image[i, :] = out_background[l,:]/max_spectra

    # Now, we need to build the observed spectral image by using:
    # - the information on the position (RA_sample, Dec_sample) for each object
    # - the information on the redshift (z_sample) for each object
    # - randomly picking up a spectrum at the good redshift among the ones built
    for ind_z, z in enumerate(z_sample):
        indices = [ind for ind, x in enumerate(out_redshift) if x == z]
        #print('found:', ind_z, z, indices)
        i = int(round((RA_sample[ind_z] - round(RA_sample[ind_z] / slice_length, 0)) / pixel_spec2, 0))
        #print('i', i)
        j = int(round(Dec_sample[ind_z] / pixel_spec1, 0))
        #print('j', j)
        k = min(67 * i + j, naxis1-1)
        #print('k', k)
        l = random.choice(indices)
        #print('l', l)
        #print(spectral_image[k, :].dtype, out_spectra[l,:].dtype)
        #print(spectral_image[k, :], out_spectra[l,:])
        spectral_image[k, :] = out_spectra[l,:]
        #print(spectral_image[k, :], out_spectra[l,:])
        #print('At row:', k, 'we insert the modelled observation', l, 'at z = ', round(z, 2),
        #                    'and (RA, Dec)=', round(RA_sample[ind_z], 4), round(Dec_sample[ind_z], 4))

        param = ' '.join(str(par) for par in out_params[l, :])
        #print(ind_z, out_params[l, :], param)
        simulation.write('%.5d %5d %.2f %.4f %.4f %s ' %(k, l, round(z, 2),
                          round(RA_sample[ind_z], 4), round(Dec_sample[ind_z], 4), param
                          ))
        simulation.write('\n')

    hdu2 = pyfits.PrimaryHDU(spectral_image[:,:])
    #hdu2.header['CTYPE1'] = 'RA--SIN'
    #hdu2.header['CUNIT1'] = 'micron'
    #hdu2.header['CRVAL1'] = 3.
    #hdu2.header['CDELT1'] = (5. - 1.)/float(naxis2)
    #hdu2.header['CRPIX1'] = 1024
    #hdu2.header['CTYPE2'] = 'RA--DEC'
    #hdu2.header['CUNIT2'] = 'arcsec'
    #hdu2.header['CRVAL2'] = 0.0
    #hdu2.header['CDELT2'] = 0.4
    #hdu2.header['CRPIX2'] = 1
    hdu2.writeto(OUT_DIR+'simobs.fits')

    simulation.close()

# AnalysisModule to be returned by get_module
Module = FLARE
