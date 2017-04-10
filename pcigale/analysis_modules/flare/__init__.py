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
import matplotlib
matplotlib.use('Agg')

import matplotlib as mpl
import matplotlib.pyplot as plt

from itertools import tee, islice, chain
import numpy as np
import numpy.ma as ma

from .. import AnalysisModule
from ..utils import backup_dir, save_fluxes

from ...utils import read_table
from .workers import init_simulation as init_worker_simulation
from .workers import simulation as worker_simulation
from ...handlers.parameters_handler import ParametersHandler

# Directory where the output files are stored
OUT_DIR = "out/"
create_simu = False

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

    #plt.show()

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

def sampleFromMF(N, alpha, logM_star, logPhi_star, logM_min, logM_max):

    import random as random
    # Draw random samples from Salpeter IMF.
    # N     ... number of samples.
    # alpha ... power-law index.
    # logM_star of the MF
    # logPhi_star of the MF
    # logM_min ... lower bound of mass interval.
    # logM_max ... upper bound of mass interval.

    # Since MF decays, maximum likelihood occurs at M_min
    #maxlik = Phi_star * (10**(0.4*(M_star-M_min)))**(alpha+1) * math.exp(-10**(0.4*(M_star-M_min))) # This is for LFs
    maxlik = np.log(10)*10**logPhi_star * 10**((logM_min-logM_star)*(1+alpha)) * np.exp(-10**(logM_min-logM_star)) # This is for MFs

    # Prepare array for output magnitudes.
    Masses = []
    # Fill in array.
    while (len(Masses) < N):
        # Draw candidate from logM interval.
        logM = random.uniform(logM_min, logM_max)

        # Compute likelihood of candidate from Salpeter SMF.
        #likelihood = Phi_star * (10**(0.4*(M_star-M)))**(alpha+1) * math.exp(-10**(0.4*(M_star-M))) # This is for LFs
        likelihood = np.log(10)*10**logPhi_star * 10**((logM-logM_star)*(1+alpha)) * np.exp(-10**(logM-logM_star)) # This is for MFs
        # Accept randomly.
        u = random.uniform(0.0, maxlik)
        if (u < likelihood):
            Masses.append(logM)

    return Masses

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
    fig, ax5 = plt.subplots(1, 1)
    cmap = mpl.cm.rainbow
    SMD = []
    m_sample = []
    z_sample = []
    N_bin_masses = 20
    N_redshifts = len(redshifts)

    #m_sample = np.zeros((N_redshifts, N_masses))

    for previous, item, next in previous_and_next(redshifts):
        if item == 0:
            previous = 0
        if previous is None:
            previous = item - min(next - item, item)
        if next is None:
            next = item + (item - previous)
        #l = FoV_axis1*(1e-3*cosmology.kpc_proper_per_arcmin(next)).value
        #h = FoV_axis2*(1e-3*cosmology.kpc_proper_per_arcmin(next)).value
        #L = (cosmology.comoving_distance(next).value-
        #        cosmology.comoving_distance(item).value)
        #volume1 = L * l * h # Volume in Mpc^2 for each redshift bin
        # Volume in Mpc**3 for the FoV
        volume = (cosmology.comoving_volume(next).value - cosmology.comoving_volume(item).value) \
                   / (4*np.pi*(180/np.pi)**2) / 3600. * FoV_axis1 * FoV_axis2
        #print('z, FoV_axis1, FoV_axis2, volume', item, FoV_axis1, FoV_axis2, volume)
        z = (item + next)/2.

        # We compute alpha, log10(M_star) and log10(Phi_star)
        alpha = func(cosmology.age(z).value * 1000., *popt1)
        logM_star = (func(cosmology.age(z).value * 1000., *popt2))
        logPhi_star = (func(cosmology.age(z).value * 1000., *popt3))
        #print('LF', item, cosmology.age(z).value * 1000., alpha, logM_star, logPhi_star)

        # We sample the stellar masses range, logarithmically
        logM = np.linspace(8., 13., N_bin_masses)

        # We compute the mass function [log10 Phi [Mpc^-3 dex^-1] for each redshift bin
        Phi_M = np.log(10)*10**logPhi_star * 10**((logM-logM_star)*(1+alpha)) * np.exp(-10**(logM-logM_star))

        #I2 = np.trapz(Phi_M, 10**logM) # integration of MF for each redshift bin per Mpc^3
        #SMD.append(np.log10(I2)) # log10 N [Mpc^-3 M_sun]
        I1 = np.trapz(Phi_M, logM) # integration of MF for each redshift bin per Mpc^3
        I2 = np.trapz(10**logM*Phi_M, logM) # integration of MF for each redshift bin per Mpc^3
        SMD.append(np.log10(I2)) # log10 N [Mpc^-3 M_sun]

        N_masses = int(I1 * volume)
        sum +=  I1 * volume
        #print('N_masses', item, I1, N_masses, sum)
        ax4.plot(logM, Phi_M, color=cmap(item/max(redshifts)))
        #ax4.set_xscale('log')
        ax4.set_yscale('log')

        #print(redshifts, item, i_redshift)
        sample = sampleFromMF(N_masses,
                              alpha, logM_star, logPhi_star,
                              np.min(logM), np.max(logM))
        if len(sample) > 0:
            m_sample = np.hstack((m_sample, sample))
            z_sample = np.hstack((z_sample, N_masses*[item]))
            #z_sample.append([N_masses*[item]])
            #print('m_sample 332', m_sample, z_sample)

        ax5.hist(sample, 100, histtype='step', color=cmap(item/max(redshifts)), lw=2, log=True)
        #ax5.plot(logM, m_sample, color=cmap(item/max(redshifts)))
        ax5.set_xlim(8, 13)

    ax4.set_xlabel('$log_{10}$ M${\star}$', fontsize=20)
    ax4.set_ylabel('$log_{10}$ Phi [Mpc$^{-3}$ dex$^{-1}$]', fontsize=20)
    ax4.text(0.6, 0.95, 'CIGALE Mass Functions from z = %.1f to z = %.1f' %(np.min(redshifts), np.max(redshifts)), ha='center', va='center', transform=ax4.transAxes)
    ax4.legend(loc=0, fontsize=6)
    ax4.grid(True)
    ax4.set_xlim(8, 13)
    ax4.set_ylim(1e-10, 1e-1)

    fig, ax6 = plt.subplots(1, 1)
    ax6.plot(redshifts, SMD)
    ax6.set_xlabel('redshift', fontsize=20)
    ax6.set_ylabel('$log_{10}$ $N$ [Mpc$^{-3}$ M$_\odot$]', fontsize=20)
    #ax6.text(0.6, 0.95, 'CIGALE Mass Functions from z = %.1f to z = %.1f' %(np.min(redshifts), np.max(redshifts)), ha='center', va='center', transform=ax4.transAxes)
    ax6.legend(loc=0, fontsize=6)
    ax5.grid(True)
    #ax6.set_xlim(1, 20)
    #ax6.set_ylim(-8, 8)

    # Plot distribution.
    #plt.show()

    # Random create of RA and Dec within the FoV
    RA_sample = [60.*random.uniform(0., FoV_axis1)
                 for ind in range(len(m_sample))]
    Dec_sample = [60.*random.uniform(0, FoV_axis2)
                for ind in range(len(m_sample))]

    return RA_sample, Dec_sample, m_sample, z_sample


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
            "cigale_string_list()",
            "List of the physical properties to save. Leave empty to save all "
            "the physical properties (not recommended when there are many "
            "models).",
            ["stellar.m_star", "attenuation.FUV", "param.FUV_luminosity"]
        )),
        ("output_file", (
            "string()",
            "Name of the output file that contains the modelled observations"
            "(photometry and spectra)",
            "cigale_sims"
        )),
        ("save_sfh", (
            "boolean()",
            "If True, save the generated Star Formation History for each model.",
            "True"
        )),
        ("FoV_axis1", (
            "float()",
            "Field ov View (arcmin)",
            1.0
        )),
        ("FoV_axis2", (
            "float()",
            "Field ov View (arcmin)",
            1.0
        )),
        ('exptime', (
            'float()',
            "Exposure time [sec]. Since FLARE photometric and spectroscopic observations"
            "are taken in parallel, we only need 1 exposure time",
            3600.0
        )),
        ("SNR", (
            "float()",
            "What is the goal for the SNR?",
            5.0
        )),
        ("S_line", (
            "float()",
            "What is the goal for S_line[erg/cm2/s]?",
            3e-18
        )),
        ("lambda_norm", (
            "float()",
            "Observed wavelength[nm] of the spectrum to which the spectrum is normalised."
            "If 0., no normalisation.",
            0.
        )),
        ("mag_norm", (
            "float()",
            "Magnitude used to normalise the spectrum at lambda_norm given above."
            "If 0., no normalisation.",
            0.
        )),
        ("create_tables", (
            "boolean()",
            "Do you want to create output tables in addition to pdf plots?",
            True
        )),
        ("flag_background", (
            "boolean()",
            "If True, save the background information "
            "for each model.",
            True
        )),
        ("flag_phot", (
            "boolean()",
            "If True, save the photometric sensitivity information"
            "for each model.",
            True
        )),
        ("flag_spec", (
            "boolean()",
            "If True, save the spectroscopic sensitivity (continuum) information"
            "for each model.",
            True
        )),
        ("flag_line", (
            "boolean()",
            "If True, save the spectroscopic sensitivity (line) information"
            "for each model.",
            True
        )),
        ("flag_sim", (
            "boolean()",
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

        redshifts = conf['sed_modules_params']['set_redshift']['redshift']
        #RA_sample, Dec_sample, z_sample = density_z(FoV_axis1, FoV_axis2, redshifts)
        #print('Echantillon', len(RA_sample), len(Dec_sample), len(z_sample))
        RA_sample, Dec_sample, m_sample, z_sample = density_m(FoV_axis1, FoV_axis2, redshifts)

        # We create FLARE spectra over 2048 pixels with 0.4x0.366-arcsec2 pixels
        slice_length = 25 # arcsec
        slice_width = 0.4 # arcsec
        pixel_spec1 = 0.366 # arcsec/pixel for spectroscopy
        pixel_spec2 = slice_width # arcsec/pixel for spectroscopy
        naxis1 = int(round(FoV_axis1 * FoV_axis2 * 3600. / (pixel_spec1 * pixel_spec2), 0))
        n_pixel2 = round(slice_length / pixel_spec2, 0)
        n_pixels = 2048
        naxis2 = n_pixels

        filters = [name for name in conf['bands'] if not
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

        if create_simu == True:
            # We do not bother to carry heavy/big arrays if we do not wish to save them.
            naxis1 = 1
            n_pixels = 2048
            naxis2 = 1

        model_spectra = (RawArray(ctypes.c_double, n_params*n_pixels),
                        (n_params, n_pixels))
        model_background = (RawArray(ctypes.c_double, naxis1*naxis2),
                        (naxis1, naxis2))
        model_redshift = (RawArray(ctypes.c_double, n_params),
                        (n_params))
        #model_masses = (RawArray(ctypes.c_double, n_params),
        #                (n_params))
        model_fluxes = (RawArray(ctypes.c_double, n_params * n_filters),
                        (n_params, n_filters))
        model_parameters = (RawArray(ctypes.c_double, n_params * n_info),
                            (n_params, n_info))

        initargs = (params, filters, info, save_sfh, create_tables, flag_background,
                    flag_phot, flag_spec, flag_line, flag_sim, create_simu,
                    lambda_norm, mag_norm,
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
        save_spectra(RA_sample, Dec_sample, m_sample, z_sample, create_simu,
                     model_spectra, model_background, model_redshift,
                     naxis1, naxis2,
                     n_params, n_pixels, model_parameters, filters, info, out_file_fits,
                     FoV_axis1, FoV_axis2, pixel_spec1, pixel_spec2)

def save_spectra(RA_sample, Dec_sample, m_sample, z_sample, create_simu,
                 model_spectra, model_background, model_redshift,
                 naxis1, naxis2,
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

    # array containing the stellar mass of each model
    #out_mass = np.ctypeslib.as_array(model_mass[0])
    #out_mass = out_mass.reshape(n_params)

    # array containing each modelled spectrum
    out_background = np.ctypeslib.as_array(model_background[0])
    out_background = out_background.reshape(naxis1, naxis2)

    out_spectra = np.ctypeslib.as_array(model_spectra[0])
    out_spectra = out_spectra.reshape(n_params, n_pixels)
    min_spectra = np.min(out_spectra)
    max_spectra = np.max(out_spectra)

    #np.set_printoptions(threshold=np.inf)
    #import ipdb; ipdb.set_trace()

    # array containing the parameters, if any, associated to each model
    out_params = np.ctypeslib.as_array(model_parameters[0])
    out_params = out_params.reshape(model_parameters[1])

    out_mass   = out_params[:, 0] # in Msun
    mask = [np.isnan(out_mass)]
    out_mass[mask] = 1.

    out_A_fuv  = out_params[:, 1]
    mask = [np.isnan(out_A_fuv)]
    out_A_fuv[mask] = 0.

    out_SFR  = out_params[:, 2]
    mask = [np.isnan(out_SFR)]
    out_SFR[mask] = 0.

    out_sSFR = (1e9 * out_SFR) / out_mass # in Gyr-1
    mask = [np.isnan(out_sSFR)]
    out_sSFR[mask]=1e-10

    out_Z  = out_params[:, 3]
    mask = [np.isnan(out_Z)]
    out_Z[mask] = 99.

    filename=OUT_DIR+'simobs.dat'
    param_names =  ' '.join(names)
    simulation = open(filename, 'w')
    simulation.write('%s %s %s' %('# Row model# redshift RA Dec mass A_fuv sSFR', param_names, '\n'))
    #simulation.write('%s %s %s' %('# Row model# redshift RA Dec ', param_names, '\n'))
    #simulation.write('# Row model# redshift RA Dec param_names \n')

    # Here, we create the simulated spectroscopic observations
    if create_simu:
        # We convert the spectra to integers [0 - 32768]
        out_spectra = out_spectra/max_spectra

        hdu = pyfits.PrimaryHDU(out_spectra[:,:])
        hdu.writeto(OUT_DIR+out_file)

        naxis1 = int(round(FoV_axis1 * FoV_axis2 * 3600. / (pixel_spec1 * pixel_spec2), 0))
        slice_length = 25 # arcsec
        n_pixel2 = round(slice_length / pixel_spec2, 0)
        naxis2 = n_pixels
        spectral_image = np.zeros((naxis1, naxis2))

        for i in range(naxis1):
            l = randint(0, naxis2-1)
            spectral_image[i, :] = out_background[l, :]/max_spectra

    # Now, we need to build the observed spectral image by:
    # - using the information on the position (RA_sample, Dec_sample) for each object
    # - using the information on the stellar mass and redshift (m_sample) for each object
    # - using the sSFT(M*, z) from Sargent et al. (2014) for each object
    # - randomly picking up a spectrum at the good mass+redshift among the ones built
    count_m    = 0
    count_afuv = 0
    count_ssfr = 0
    count_zsma = 0
    for ind_z, z in enumerate(z_sample):

        dm = 0.2
        m_mf = 10**m_sample[ind_z]
        # New 2D formula: A_fuv_mf(m_mf, z)
        #fprint('m_mf', m_mf)
        if np.log10(m_mf) > 7.0:

            #v1
            #A_fuv_mf = (0.05 +
            #            0.25 * np.exp(-(((z-2.0)/1.5)**2)) +
            #            0.03 * np.exp(-(((z-4.5)/2.0)**2))  ) * (np.log10(m_mf)-7)**2

            #v2
            #a1 = 0.1
            #m1 = 1.0
            #sigma1 = 0.5
            #a2 = 0.3
            #m2 = 2.5
            #sigma2 = 1.6
            #A_fuv_mf = (a1 * np.exp(-(z-m1)**2/(2*sigma1**2)) + \
            #            a2 * np.exp(-(z-m2)**2/(2*sigma2**2))) * \
            #           (np.log10(m_mf)-7)**2

            #v3
            #a = 0.17
            #b = 0.21
            #c = 1.77
            #d = 2.19

            #v4
            #a = 0.086139429
            #b = 0.095825209
            #c = 2.31819179
            #d = 1.54701983

            #v5
            #a = 0.10078457 # 0.09554179
            #b = 0.03006847 # 0.03775311
            #c = 4.48576362 # 3.85981481
            #d = 4.95338566 # 5.06978438

            #A_fuv_mf = ((a + b*z) / (1 + (z/c)**d)) * (np.log10(m_mf)-7)**2

            #v6: 2 gaussians
            #a1     = 0.171
            #m1     = 1.132
            #sigma1 = 1.066
            #a2     = 0.166
            #m2     = 3.648
            #sigma2 = 1.032

            #v7 2 gaussians
            #a1     = 0.049
            #m1     = 0.812
            #sigma1 = 0.294
            #a2     = 0.242
            #m2     = 2.172
            #sigma2 = 2.019

            #v8 2 gaussians with Bouwens + evolving Td
            a1     = 0.1249
            m1     = 0.7690
            sigma1 = 0.7466
            a2     = 0.2375
            m2     = 3.1601
            sigma2 = 1.8710

            A_fuv_mf = (a1 * np.exp(-(z-m1)**2/(2*sigma1**2)) + \
                        a2 * np.exp(-(z-m2)**2/(2*sigma2**2))) * \
                       (np.log10(m_mf)-7)**2

        else:
            A_fuv_mf = 0.025*np.log10(m_mf)

        # We compute the sSFR from M* and CIGALE's SFR10Myrs
        # sSFR(M, z) = N(M) exp (A·z / (1 + B·z^C) ) in Gyr-1
        # where
        # N(M, z) = N(5e10 Msun) 10^ν log(M*/[5e10 Msun])
        # N(5e10 Msun) = 0.095 +0.002/−0.003 Gyr−1
        # ν ~ −0.2
        # A = 2.05 +0.33/−0.20
        # B = 0.16 +0.15/−0.07
        # C = 1.54 +0.32/-0.32

        A = 2.05
        B = 0.16
        C = 1.54
        nu = -0.21
        m_ref = 5e10
        # v2 onlys uses Sargent et al. (2014)
        # v2 & v4 use Sargent et al. (2014, Fig. 18),         valid above logMstar = 10.2
        #         use Whitaker Sargent et a. (2014; Fig/ 17), valid below logMstar = 10.2
        if np.log10(m_mf) > 10.2:
            # This is from Sargent et al. (2014, Fig. 18), valid above logMstar = 10.2
            N = 0.095
            sSFR_mf = N * 10**(nu * np.log10(m_mf/m_ref)) * np.exp(A*z / (1+B*z**C))
        else:
            # This is from Whitaker Sargent et a. (2014; Fig/ 17), valid below logMstar = 10.2
            N = 5.994e-12
            sSFR_mf = N * 10**(np.log10(m_mf)+nu * np.log10(m_mf/m_ref)) * np.exp(A*z / (1+B*z**C))

        # We select the models with the requested stellar mass
        i_m_best, m_best = min(enumerate(out_mass), key=lambda x: abs(x[1]-m_mf))
        #print('i_m_best, m_best, m_mf, A_fuv_mf', i_m_best, m_best, m_mf, A_fuv_mf)
        # If the requested stellar mass is not in out_mass, we take the closest one
        if (m_best < m_mf*(1.-dm) or m_best > m_mf*(1.+dm)):
            count_m += 1
            #print('count_m', count_m, m_best, m_mf)
        m = m_best

        # We select the models with the requested dust attenuation
        dA_fuv = 0.2
        j_A_fuv_best, A_fuv_best = min(enumerate(out_A_fuv), key=lambda x: abs(x[1]-A_fuv_mf))
        if (A_fuv_best < A_fuv_mf*(1.-dA_fuv) or A_fuv_best > A_fuv_mf*(1.+dA_fuv)):
            count_afuv += 1
            #print('count_afuv', count_afuv, A_fuv_best, A_fuv_mf, out_A_fuv)
        A_fuv = A_fuv_best

        # We select the models with the requested specific star formation rate
        dsSFR = 0.2
        k_sSFR_best, sSFR_best = min(enumerate(out_sSFR), key=lambda x: abs(x[1]-sSFR_mf))
        #print('sSFR', ind_z, z, k_sSFR_best, sSFR_best, sSFR_mf)
        # If the requested sSFR is not in out_sSFR, we take the closest one
        if (np.isnan(sSFR_best) or sSFR_best < sSFR_mf*(1.-dsSFR) or sSFR_best > sSFR_mf*(1.+dsSFR)):
            count_ssfr += 1

        sSFR = sSFR_best

        # We create a mask for models with the good redshift and good stellar mass
        dz = 0.1*z
        #mask_zsma = (out_redshift >= z*(1.-dz))      & (out_redshift <= z*(1.+dz))    & \
        #            (out_mass  >= m*(1.-dm))         & (out_mass  <= m*(1.+dm))       & \
        #            (out_sSFR >= sSFR*(1.-dsSFR))    & (out_sSFR <= sSFR*(1.+dsSFR))  & \
        #            (out_A_fuv >= A_fuv*(1.-dA_fuv)) & (out_A_fuv <= A_fuv*(1.+dA_fuv))

        #np.set_printoptions(threshold=np.inf)
        #import ipdb; ipdb.set_trace()

        mask_zsma = \
          (np.abs(out_redshift-z)/z      <= dz)     & \
          (np.abs(out_mass-m)/m          <= dm)     & \
          (np.abs(out_sSFR-sSFR)/sSFR    <= dsSFR)  & \
          (np.abs(out_A_fuv-A_fuv)/A_fuv <= dA_fuv)

        masked_indx = np.where(mask_zsma)

        if len(masked_indx[0])==0:
            #print('No model found, we skip it')
            count_zsma += 1
            params = str(m)+' '+str(A_fuv)+' '+str(sSFR)+' '+' nan'*(model_parameters[1][1])
            simulation.write('%.5d %5d %.2f %.4f %.4f %s ' %(-1, -1, round(z, 2),
                             round(RA_sample[ind_z], 4),
                             round(Dec_sample[ind_z], 4),
                             params
                             ))
            simulation.write('\n')
        else:
        #print('0.m>', out_mass == m)
        #print('0.z>', out_redshift == z)
        #print('0.A_fuv>', (out_A_fuv >= A_fuv*(1.-dA_fuv)) & (out_A_fuv <= A_fuv*(1.+dA_fuv)))
        #print('1>', m, z, A_fuv, mask_zma, out_mass[mask_zma], out_redshift[mask_zma], out_A_fuv[mask_zma])
        #print('2>', masked_indx)
        #print('3>', out_mass[masked_indx])
        #print('4>', out_redshift[masked_indx])
            indx = masked_indx[0][np.random.randint(0, len(masked_indx[0]), size=1)]

            if create_simu:

                i = int(round((RA_sample[ind_z] - round(RA_sample[ind_z] / slice_length, 0)) / pixel_spec2, 0))
                j = int(round(Dec_sample[ind_z] / pixel_spec1, 0))
                k = min(67 * i + j, naxis1-1)

                spectral_image[k, :] = out_spectra[indx, :]
            else:
                k = ind_z
            #print('At row:', k, 'we insert the modelled observation', indx, 'at z = ', round(z, 2),
            #                    'and (RA, Dec)=', round(RA_sample[ind_z], 4), round(Dec_sample[ind_z], 4))

            params = str(m)+' '+str(A_fuv)+' '+str(sSFR)+' '
            params = params + ' '.join(str(par) for par in out_params[indx, :][0])
            simulation.write('%.5d %5d %.2f %.4f %.4f %s ' %(k, indx, round(z, 2),
                             round(RA_sample[ind_z], 4),
                             round(Dec_sample[ind_z], 4),
                             params
                             ))
            simulation.write('\n')

    print('WARNING: for', round(100*count_m/len(z_sample), 1), '% of the sources', \
          'we found no modelled galaxy M* '\
          'within a factor of 2, and, we picked the closest one.')
    print('WARNING: for', round(100*count_ssfr/len(z_sample), 1), '% of the sources', \
          'we found no modelled galaxy sSFR '\
          'within +/-', 100.*dsSFR, '%, and, we picked the closest one.')
    print('WARNING: for', round(100*count_afuv/len(z_sample), 1), '% of the sources', \
          'we found no modelled galaxy A_fuv '\
          'within +/-', 100.*dA_fuv, '%, and, we picked the closest one.')
    print('WARNING: for', round(100*count_zsma/len(z_sample), 1), '% of the sources', \
          'we found no model galaxy, and, we skipped them.')

    if create_simu:
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
