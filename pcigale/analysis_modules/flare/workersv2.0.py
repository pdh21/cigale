# -*- coding: utf-8 -*-
# Copyright (C) 2016 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Denis Burgarella

"""
Module that simulates observations with the FLARE telescope
==========================================================================

This module comes after a traditional CIGALE run, in the modelling version.

It uses the modelled spectra and simulates observations assuming he characteristics
of FLARE in terms of size of the optics, temperature, wavelength range, detectors,
thermal and all background noises.

To modify the characteristics of FLARE, you need to edit this file

The simulated observations with will created, either photometry or spectroscopy.

Normally, this module will come at the end of the module list after all the models
have been created.

"""

import time

import numpy as np

from ...warehouse import SedWarehouse
from ..utils import OUT_DIR

from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib as mpl
import sys, os.path
import os.path

from . import AnalysisModule
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import resample
from pcigale.sed.utils import lambda_to_nu
from pcigale.sed.utils import luminosity_to_flux
from pcigale.sed.utils import lambda_flambda_to_fnu
import pcigale.sed
from ...data import Database


#----------------------------------------------------------------------
def set_config(instrument, wave):
    """
    Defines which configuration/instrument, we simulate
    """
    if instrument == 'FLARE':
        #-------------FLARE

        # Physical and mathematical constants
        h = 6.626e-34 # J.s or J/Hz
        c = 3.0e8    # m/s
        k = 1.38e-23  # J/K
        pi = 3.1415926

        # For FLARE: same exposure time for imaging and spectroscopy
        # We assume that FLARE is diffraction-limited:
        #Diff_Limit = 0.354 # 2.5um for 1.5m ~ 0.354 arcsec FWHM (1.03*lambda/D)
        #Diff_Limit = 0.354 # 3.0um for 1.8m ~ 0.354 arcsec FWHM (1.03*lambda/D)
        #Diff_Limit = 0.372 # 3.4um for 2.0m ~ 0.342 arcsec FWHM (1.03*lambda/D)
        #First_min = 0.420 # arcsec @ 2.5um for 1.5m and @ 3.0um for 1.8m
        First_min = 0.441 # arcsec @ 3.4um for 2.0m
        D1 = 200 # cm
        D2 = 0.187 * D1 # Same assumption as WISH
        A = pi*((D1/2)**2 - (D2/2)**2) # cm2
        pixel_ima = 0.15 # arcsec/pixel # For imaging
        eta_tel = 0.97**4 # 0.885
        strehl = 0.90 # For JWST, Strehl ratio is the ratio of peak diffraction
                      # intensities of an aberrated vs. perfect wavefront.
        EE2um = 0.838 * strehl # Encircled Energy of 83.8% at First_min = 1.22*lambda/D
            # From http://www.telescope-optics.net/diffraction_image.htm#The_diameter_

        Omega_ima = (1 + np.rint(First_min / pixel_ima))**2 # a squared area in pixels
        R_ima = 4 # lambda / Delta_lambda
        slice = 0.4 # arcsec ~ 1 pixel
        R_spec = 750 # lambda / Delta_lambda
        #pixel_spec = 0.366 # arcsec/pixel for spectroscopy
        pixel_spec = slice # arcsec/pixel for spectroscopy
        Omega_spec = pixel_spec * 2 # 1 slice: 1 pixel x 2 pixels per spectral res. elt
        eta_spec = 0.97**10 * 0.70  # 0.737 * 0.70 = 0.516
        emissivity = 0.05 # emissivity
        T_tel_min = 80 # K
        T_tel_max = 90 # K
        Delta_T_tel = 2 # K
        onexposure_ima = 500 # sec
        onexposure_spec = 500 # sec

    elif instrument == 'JWST':
        #---------------JWST

        # For JWST: same exposure time for imaging and spectroscopy
        D1 = 564 # cm rom http://jwst.nasa.gov/faq_scientists.html
        D2 = 0.11 * D1 # From http://jwst.nasa.gov/faq_scientists.html
        A = pi*((D1/2)**2 - (D2/2)**2)
        pixel_ima = 0.032  # arcsec/pixel which is the average
                   # of 0.032 for JWST/NIRCAM short and 0.065 for JWST/NIRCAM long
        pixel_spec = 0.1 # arcsec/pixel for spectroscopy
        slice = 0.1 # arcsec/pixel for JWST/NIRSPEC
        eta_tel = 0.65 #
        strehl = 0.80 # Strehl ratio is the ratio of peak diffraction intensities
                  # of an aberrated vs. perfect wavefront.
        EE2um = 0.838 * strehl # Encircled Energy: 83.8% inside r=2.5*0.032 ~1.22*lambda/D
        R_ima = 4 # lambda / Delta_lambda
        Omega_ima = 2.5**2 # in pixel for JWST/NIRCam
        R_spec = 1000 # lambda / Delta_lambda
        Omega_spec = 2.5 * 2 # 1 slice, i.e., pixel in space x 2 pixels / spect. res. elt
        eta_spec = 0.6 #
        emissivity = 0.05 # emissivity
        T_tel_min = 60 # K
        T_tel_max = 100 # K
        Delta_T_tel = 10 # K
        onexposure_ima = 10 # sec
        onexposure_spec = 500 # sec

    return onexposure_ima, onexposure_spec, pixel_ima, pixel_spec, \
           slice, eta_tel, EE2um, Omega_ima, A, R_ima, R_spec, Omega_spec, eta_spec, \
           T_tel_min, T_tel_max, Delta_T_tel, emissivity

#
#----------------------------------------------------------------------
def blackbody_lam(wave, T):
    """
    Creates a blackbody with temperature T

    input: wavelength [m]
           temperature [K]
    output: blackbody [erg/s/cm2/um/arcsec2]

    """
    # Physical and mathematical constants
    h = 6.626e-34 # J.s or J/Hz
    c = 3.0e8    # m/s
    k = 1.38e-23  # J/K
    pi = 3.1415926

    #BB = 2 * h * c**2 / wave**5 / (np.exp(h*c / (wave*k*T)) - 1) # W/m3/sr
    steradian = 4.25e10 # arcsec2
    # W/m2/m/sr = (1e7*erg/s)/(1e4*cm2)/(1e6*um)/(4.25e10*sr)
    #           = 1e7/1e4/1e6/4.25e10 erg/s/cm2/um/arcsec2
    #           = 1e-3/4.25e10 erg/s/cm2/um/arcsec2

    return 2e-3 * h * c**2 / (wave**5 * (np.exp(h*c / (wave*k*T)) - 1)) / steradian
#
#----------------------------------------------------------------------
def ZodiacalLight(wave):
    """
    Two possibilities to create the Zodi:

    a) scattering at 5800K and thermal at 280K
    From:http://home.strw.leidenuniv.nl/~franx/nirspec/sensitivity/NIRSpec_sens_Rev2.0.pdf
    Zodi(lambda[um]) = 2.0e8*(wave[μm])**(-1.8) + 7.0e-8*BB(T=256K) [photons/s/cm2/μm/sr]
    where BB(T) is the black body at T = 256K (in appropriate photon units)
    The first term in is the contribution due to scattered solar radiation
    The second term is the (diluted) thermal emission from the zodiacal dust.

    b) open Leinert's (1998) Zodi in erg/s/cm2/A/arcsec2
    http://aas.aanda.org/articles/aas/pdf/1998/01/ds1449.pdf Table 19
    and re-sample the table to out wavelength table: wave

    input: wavelength [m]
    output: Zodi_from_BB [erg/s/cm2/A/arcsec2]
    """

    # Physical and mathematical constants
    h = 6.626e-34 # J.s or J/Hz
    c = 3.0e8    # m/s
    k = 1.38e-23  # J/K
    pi = 3.1415926
    steradian = 4.25e10 # arcsec2 / sr
    joule = 1e7 # J / erg
    nu = c / wave # Hz
    E_photon = h * nu # Joule / photon
    BB256  = blackbody_lam(wave, 256.) / joule / E_photon * steradian
    BB5800 = blackbody_lam(wave, 5800) / joule / E_photon * steradian
    # above BB in units of erg/s/cm2/um/arcsec2 -> photons/s/cm2/μm/sr
    #Leinert = np.genfromtxt('Zodi_Leinert1998.dat', dtype=('f4', 'f4'), \
    #          names = True, missing_values = ('nn'), filling_values=(np.nan))
    #wave_Leinert = Leinert['wave']
    #Zodi_Leinert = Leinert['zodi'] # in erg/s/cm2/A/arcsec2
    #f = interp1d(wave_Leinert/1e10, Zodi_Leinert)
    #Zodi_from_Leinert = f(wave) # in erg/s/cm2/A/arcsec2

    # in units of erg/s/cm2/um/arcsec2 -> photons/s/cm2/μm/sr
    Zodi_from_BB_Jakobsen = 2.0e8*(wave*1e6)**(-1.8) + 7.1e-8*BB256 # in ph/s/cm2/μm/sr
    # From Jakobsen's note:
    # http://home.strw.leidenuniv.nl/~franx/nirspec/sensitivity/NIRSpec_sens_Rev2.0.pdf
    Zodi_from_BB_Allen = 3e-14 * BB5800 + 7.0e-8 * BB256 # in ph/s/cm2/μm/sr
    # From Allen's book, p. 146: https://books.google.nl/books?id=w8PK2XFLLH8C&redir_esc=y

    Zodi_from_BB_Jakobsen = Zodi_from_BB_Jakobsen * E_photon * joule / steradian / 1e4
    # in units of ph/s/cm2/μm/sr -> erg/s/cm2/A/arcsec2
    Zodi_from_BB_Allen = Zodi_from_BB_Allen * E_photon * joule / steradian / 1e4
    # in units of ph/s/cm2/μm/sr -> erg/s/cm2/A/arcsec2

    # Now, we need to select which spectrum of the zodiacal light we want:
    #Zodi = Zodi_from_Leinert
    Zodi = Zodi_from_BB_Jakobsen
    #Zodi = Zodi_from_BB_Allen

    return(Zodi)
#
#----------------------------------------------------------------------
def Detector(instrument, wave):
    """
    input: wavelength [m]
    output:
    - RON [e-]
    - Dark [e-/sec]
    - QE []
    """
    if instrument == 'FLARE':
        RON = 15. * np.ones_like(wave) # e-
        Dark_Temp = np.array([35., 40., 50., 60., 70., 80., 90., 100.]) # in K
        Dark_teledyne = np.array([0.0032, 0.005, 0.01, 0.032, 0.1, 1., 31.2,  1000.]) # in e-/pix/sec
        T_detector = 60 #K
        Dark = np.interp(T_detector, Dark_Temp, Dark_teledyne) * np.ones_like(wave) # e-/sec
        #Dark = 0.05 * np.ones_like(wave) # e-/sec
        QE = 0.75 * np.ones_like(wave)
        # from http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20140008961.pdf
        ADU = 1.53 # e-/ADU
    elif instrument == 'JWST':
        RON = 6. * np.ones_like(wave) #e- for JWST NIRCAM & NIRSPEC
        Dark = 0.01 * np.ones_like(wave) # e-/sec for JWST  NIRCAM & NIRSPEC
        QE = 0.80 * np.ones_like(wave)
        # from http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20140008961.pdf
        ADU = 1.53 # e-/ADU

    return(RON, Dark, QE, ADU)
#
#----------------------------------------------------------------------
def ThermalRadiation(wave, T_telescope):
    """
    Thermal radiation from the telescope assumed at T_telescope
    in units of erg/s/cm2/A/arcsec2

    input: wavelength [m]
           temperature telescope [K]
    output: Zodi_from_BB [erg/s/cm2/A/arcsec2]
    """

    Thermal  = blackbody_lam(wave, T_tel) / 1e4
    # above BB in units of erg/s/cm2/A/arcsec2

    return(Thermal)
#
#----------------------------------------------------------------------
def PlotBackground(Project, wave, Zodi, Thermal, T_tel, redshift, idx, create_table):
    """
    Plots the backgrounds

    inputs:
    - wave [m]
    - Zodi [erg/cm2/s/A/arcsec2]
    - Thermal [erg/cm2/s/A/arcsec2]
    - T_tel [K]
    """

    # Directory where the output files are stored
    OUT_DIR = "out/background/"

    # Physical and mathematical constants
    h = 6.626e-34 # J.s or J/Hz
    c = 3.0e8    # m/s
    k = 1.38e-23  # J/K
    pi = 3.1415926

    fig1 = plt.figure(figsize=(8, 10))
    fig1.canvas.set_window_title('Background')
    ax1 = fig1.add_subplot(111)

    # baseline units are in erg/s/cm2/A/arcsec$2
    # To convert to units in MJy/sr
    # 1 erg/s/cm2/Hz/arcsec2 = 1 erg/s/cm2/A/arcsec2 * (wave**2/c)
    # 1 Jy/arcsec2 = 1 erg/s/cm2/Hz/arcsec2 * 1e23
    # 1 MJy/arcsec2 = 1 Jy/arcsec2 / 1e6
    # 1 MJy/sr = 1 MJy/arcsec2 * 4.25e10
    Conv_to_MJypersr = (1e10*(wave**2/c)) * 1e23 / 1e6 * 4.25e10
    #
    # Comparison in MJy/sr with JWST's background:
    # http://jwst.nasa.gov/resources/JWST_StatusPlan_111216.pdf
    # Wave_JWST = [1.0, 2.0, 3.0, 3.5, 4.0, 5.0] # in microns
    # Bkgd_JWST_mean = [0.120, 0.070,  0.040, 0.050, 0.090, 0.300]  # in MJy/sr
    # Bkgd_JWST_NEP  = [0.040, 0.040,  0.020, 0.019, 0.018, 0.055]  # in MJy/sr
    # Bkgd_JWST_HUDF = [0.032, 0.030,  0.013, 0.014, 0.016, 0.050]  # in MJy/sr
    #
    # plt.plot(Wave_JWST, Bkgd_JWST_HUDF, 'or', linestyle=':', label="Bkgd_JWST_HUDF")
    # plt.plot(Wave_JWST, Bkgd_JWST_NEP,  'sr', linestyle=':', label="Bkgd_JWST_NEP")
    # plt.plot(Wave_JWST, Bkgd_JWST_mean,  '^b', linestyle='--', label="Bkgd_JWST_mean")

    ax1.plot(wave*1e6, Zodi[:], 'k', label="Zodiacal light")
    cmap = mpl.cm.jet_r
    for i_T_tel, T_tel_i in enumerate(T_tel):
        ax1.plot(wave*1e6, Zodi[:] + Thermal[i_T_tel, :],
                 label='Background @ '+str(T_tel_i)+'K',
                 color=cmap(i_T_tel / float(len(T_tel))))

    ax1.set_xlabel('Observed wavelength ($\mu m)$', fontsize=20)
    ax1.set_ylabel('$erg/s/cm^2/A/arcsec^2$', fontsize=20)
    ax1.legend(loc=0, fontsize=8)
    ax1.set_xlim(1, 5)
    ymin = 0.9*np.min(Zodi[:] + Thermal[i_T_tel, :])
    ymax = 1.1*np.max(Zodi[:] + Thermal[i_T_tel, :])
    ax1.set_ylim(ymin, ymax)
    #ax1.set_yscale('log')
    ax4 = ax1.twiny()
    ax4.set_xlim(1./(1+redshift), 5./(1+redshift))
    ax4.set_xlabel('Rest-frame Wavelength [$\mu m$]', fontsize=20)

    ax2 = ax1.twinx()
    ax2.set_ylabel('$MJy/sr$', fontsize=20)
    ax2.set_xlim(1, 5)
    ymin = 0.9*np.min((Zodi[:] + Thermal[i_T_tel, :])*Conv_to_MJypersr[:])
    ymax = 1.1*np.max((Zodi[:] + Thermal[i_T_tel, :])*Conv_to_MJypersr[:])
    ax2.set_ylim(ymin, ymax)
    fig1.suptitle(Project+'\'s backgrounds', fontsize=20)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    if create_table == True:
        filename=OUT_DIR+str(idx)+'_background.txt'
        background = open(filename, 'w')
        background.write('# Wave Zodi Thermal Total T_tel \n')
        for i_T_tel, T_tel_i in enumerate(T_tel):
            for iwave, wavelength in enumerate(wave):
                background.write('%.5f %2s' %(wavelength*1e6, '  '))
                background.write('%.2e %2s' %(Zodi[iwave], '  '))
                background.write('%.2e %2s' %(Thermal[i_T_tel, iwave], '  '))
                background.write('%.2e %2s' %(Zodi[iwave]+Thermal[i_T_tel, iwave], '  '))
                background.write('%.1f %2s' %(T_tel_i, '  '))
                background.write('\n')
        background.close()

    fig1.savefig(OUT_DIR + str(idx) + '_background.pdf')
    plt.close(fig1)

    #plt.show()
#
#----------------------------------------------------------------------
def PlotPhot(S_nu_mAB, S_nu_SNR, SNR_mAB, lambda_min, lambda_max, lambda_eff,
             FLARE_filters, FLARE_fluxes, FLARE_mAB, FLARE_SNR,
             T_tel, redshift, exptime, R_ima, idx, create_table):
    """
    Plots photometric SNR_phot for input spectrum and exposure time.
    Plots f_nu / mAB for a given SNR_mAB and exposure time.

    Inputs:
    - wave [m]
    - SNR_phot []
    - S_nu_SNR []
    - filters: FLARE filter names
    - lambda_eff: Effective wavelength of FLARE filters [m]
    - fluxes: Flux densities in FLARE filters [erg/s/cm2/Hz]
    - T_tel [K]

    """
    # Directory where the output files are stored
    OUT_DIR = "out/phot/"
    Lyman_alpha = 0.1216

    fig2 = plt.figure(figsize=(8, 10))
    fig2.canvas.set_window_title('Photometric SNR')
    cmap = mpl.cm.jet_r
    ax1 = fig2.add_subplot(211)
    ax2 = fig2.add_subplot(212)
    for i_T_tel, T_tel_i in enumerate(T_tel):
        ax1.errorbar(1e-3*lambda_eff, FLARE_SNR[i_T_tel, :], xerr=1e-3*lambda_eff/R_ima/2.,
           fmt='o', ecolor=cmap(i_T_tel / max(1, float(len(T_tel)-1))),
           label='SNR_phot for input CIGALE spectrum'+
           ' & Background @ '+str(T_tel_i)+'K @ z ='+str(round(redshift,2)),
           color=cmap(i_T_tel / max(1, float(len(T_tel)-1))), linestyle='None')
        for i_band in range(len(lambda_eff)):
            if FLARE_SNR[i_T_tel, i_band] > 0.01:
                ax1.annotate(FLARE_filters[i_band]+'='+
                         str(round(FLARE_mAB[i_band], 1)),
                     xy=(+0.10+1e-3*lambda_eff[i_band], FLARE_SNR[i_T_tel, i_band]),
                     rotation=0, size=8)
    ax1.set_xlim(0.9, 5.1)
    ymin = max(np.mean(FLARE_SNR[i_T_tel, :])-3.*np.std(FLARE_SNR[i_T_tel, :]),0)
    ymax = np.mean(FLARE_SNR[i_T_tel, :])+3.*np.std(FLARE_SNR[i_T_tel, :])
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlabel('Observed Wavelength [$\mu m$]', fontsize=20)
    ax1.set_ylabel('$SNR_{phot}$', fontsize=20)
    ax1.legend(loc=0, fontsize=8, numpoints = 1)
    #ax1.set_yscale('log')
    ax4 = ax1.twiny()
    ax4.set_xlim(1./(1+redshift), 5./(1+redshift))
    ax4.set_xlabel('Rest-frame Wavelength [$\mu m$]', fontsize=20)

    for i_T_tel, T_tel_i in enumerate(T_tel):
        ax2.errorbar(1e-3*lambda_eff, S_nu_SNR[i_T_tel,:], xerr=1e-3*lambda_eff/R_ima/2.,
                fmt='o', ecolor=cmap(i_T_tel / max(1, float(len(T_tel)-1))),
                label='Snu_phot for SNR = '+str(SNR_mAB)+
                ' & Background @ '+str(T_tel_i)+'K & T_exp='+str(exptime)+' sec',
                color=cmap(i_T_tel / max(1, float(len(T_tel)-1))), linestyle='None')
    ax2.set_xlim(0.9, 5.1)
    ymin = max(np.mean(S_nu_SNR[i_T_tel, :])-2.*np.std(S_nu_SNR[i_T_tel, :]), 1.)
    ymax = np.mean(S_nu_SNR[i_T_tel, :])+3.*np.std(S_nu_SNR[i_T_tel, :])
    ax2.set_ylim(ymin, ymax)
    ax2.set_xlabel('Observed Wavelength $\mu m$', fontsize=20)
    ax2.set_ylabel('S$_{nu} [nJy]$', fontsize=20)
    #ax2.set_yscale('log')
    ax2.legend(loc=0, fontsize=8, numpoints=1)

    ax3 = ax2.twinx()
    ax3.set_xlim(0.9, 5.1)
    ax3.set_ylabel('m$_{AB}$', fontsize=20)
    ymin = -2.5*np.log10(ymin)+31.4
    ymax = -2.5*np.log10(ymax)+31.4
    ax3.set_ylim(ymin, ymax)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    if create_table == True:

        filename=OUT_DIR+str(idx)+'_Photometry.txt'
        photometry = open(filename, 'w')
        photometry.write('# Wave SNR_phot m_AB f_nu T_tel \n')

        for ifilt in range(len(FLARE_filters)):
            photometry.write('%.5f %2s' %(lambda_eff[ifilt]*1e6, '  '))
            photometry.write('%.2f %2s' %(FLARE_SNR[i_T_tel, ifilt], '  '))
            photometry.write('%.2f %2s' %(-2.5*np.log10(FLARE_fluxes[ifilt])-48.56, '  '))
            photometry.write('%.2f %2s' %(FLARE_fluxes[ifilt]*1e32, '  '))
            photometry.write('%.1f %2s' %(T_tel_i, '  '))
            photometry.write('\n')

        photometry.close()

    fig2.savefig(OUT_DIR + str(idx) + '_phot.pdf')
    plt.close(fig2)

    #plt.show()
#
#----------------------------------------------------------------------
def PlotSpec(wave, SNR_spec, SNR_mAB, mAB_SNR, S_nu_SNR, T_tel,
             redshift, exptime, idx, create_table):

    """
    Plots spectroscopic SNR_spec for input spectrum and exposure time.
    Plots f_nu / mAB for a given SNR_mAB and exposure time.

    Inputs:
    - wave [m]
    - SNR_spec []
    - mAB_SNR [AB mag]
    - S_nu_SNR []
    - T_tel [K]

    """
    # Directory where the output files are stored
    OUT_DIR = "out/cont/"

    fig3 = plt.figure(figsize=(8, 10))
    fig3.canvas.set_window_title('Spectroscopic SNR')
    cmap = mpl.cm.jet_r
    ax1 = fig3.add_subplot(211)
    ax2 = fig3.add_subplot(212)
    for i_T_tel, T_tel_i in enumerate(T_tel):
         ax1.plot(wave*1e6, SNR_spec[i_T_tel, :],
            label='SNR_spec for input CIGALE spectrum'+
            ' & Background @ '+str(T_tel_i)+'K @ z ='+
            str(round(redshift, 2))+' & T_exp='+str(exptime)+' sec',
            color=cmap(i_T_tel / max(1, float(len(T_tel)-1))))
    ax1.set_xlim(1, 5)
    #ax1.set_ylim(0.5*min(SNR_spec[i_T_tel, :]), max(2.*SNR_spec[i_T_tel, :]))
    ymin = max(np.mean(SNR_spec[i_T_tel, :])-3.*np.std(SNR_spec[i_T_tel, :]),0)
    ymax = np.mean(SNR_spec[i_T_tel, :])+3.*np.std(SNR_spec[i_T_tel, :])
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlabel('Observed Wavelength [$\mu m$]', fontsize=20)
    ax1.set_ylabel('SNR$_{spec}$', fontsize=20)
    ax1.legend(loc=0, fontsize=8)
    #ax1.set_yscale('log')
    ax4 = ax1.twiny()
    ax4.set_xlim(1./(1+redshift), 5./(1+redshift))
    ax4.set_xlabel('Rest-frame Wavelength [$\mu m$]', fontsize=20)

    for i_T_tel, T_tel_i in enumerate(T_tel):
        ax2.plot(wave*1e6, S_nu_SNR[i_T_tel, :],
                label='Snu_spec for SNR = '+str(SNR_mAB[0])+
                ' & Background @ '+str(T_tel_i)+'K & T_exp='+str(exptime)+' sec',
                color=cmap(i_T_tel / max(1, float(len(T_tel)-1))))
    ax2.set_xlim(1, 5)
    #ax2.set_ylim(0.5*min(S_nu_SNR[i_T_tel, :]), max(2.0*S_nu_SNR[i_T_tel, :]))
    ymin = max(np.mean(S_nu_SNR[i_T_tel, :])-3.*np.std(S_nu_SNR[i_T_tel, :]),0)
    ymax = np.mean(S_nu_SNR[i_T_tel, :])+3.*np.std(S_nu_SNR[i_T_tel, :])
    ax2.set_ylim(ymin, ymax)
    ax2.set_xlabel('Observed Wavelength [$\mu m$]', fontsize=20)
    ax2.set_ylabel('S$_{nu}$ [$nJy$]', fontsize=20)
    #ax2.set_yscale('log')
    ax2.legend(loc=0, fontsize=8)

    ax3 = ax2.twinx()
    ax3.set_xlim(1, 5)
    ax3.set_ylabel('m$_{AB}$', fontsize=20)
    ax3.set_ylim(-2.5*np.log10(0.99*min(S_nu_SNR[i_T_tel, :]))+31.4,
                 -2.5*np.log10(1.01*max(S_nu_SNR[i_T_tel, :]))+31.4)

    #for i_T_tel, T_tel_i in enumerate(T_tel):
    #    for iwave, wavelength in enumerate(wave):
    #        logging.info('For SNR_mAB=%.2f, Wave=%.2f um, S_nu=%.1f nJy'%(
    #           SNR_mAB[0], wavelength*1e6, S_nu_SNR[i_T_tel, iwave]))
    #logging.info('--------------------------')

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    if create_table == True:

        filename=OUT_DIR+str(idx)+'_spectro_cont.txt'
        continuum = open(filename, 'w')
        continuum.write('# Wave SNR_mAB S_nu_SNR T_tel \n')

        for i_T_tel, T_tel_i in enumerate(T_tel):
            for iwave, wavelength in enumerate(wave):
                continuum.write('%.5f %2s' %(wavelength*1e6, '  '))
                continuum.write('%.2f %2s' %(SNR_mAB[0], '  '))
                continuum.write('%.2e %2s' %(S_nu_SNR[i_T_tel, iwave], '  '))
                continuum.write('%.1f %2s' %(T_tel_i, '  '))
                continuum.write('\n')

        continuum.close()

    fig3.savefig(OUT_DIR + str(idx) + '_continuum.pdf')
    plt.close(fig3)

    #plt.show()
#----------------------------------------------------------------------
def PlotLine(wave, S_line, SNR_line, SNR_flambda, mAB_SNR, S_line_SNR, T_tel,
             redshift, exptime, idx, create_table):

    """
    Plots photometric SNR_line for input spectrum and exposure time.
    Plots f_nu / mAB for a given SNR_mAB and exposure time.

    Inputs:
    - wave [m]
    - SNR_line []
    - mAB_SNR [AB mag]
    - S_line_SNR []
    - T_tel [K]

    """
    # Directory where the output files are stored
    OUT_DIR = "out/line/"

    fig4 = plt.figure(figsize=(8, 10))
    fig4.canvas.set_window_title('Spectroscopic Line SNR')
    cmap = mpl.cm.jet_r
    ax1 = fig4.add_subplot(211)
    ax2 = fig4.add_subplot(212)
    for i_T_tel, T_tel_i in enumerate(T_tel):
        ax1.plot(wave*1e6, SNR_line[i_T_tel, :],
                 label='SNR_line for S_line = '+str(S_line)+
                 '[erg/cm$^2$/s] & Background @ '+str(T_tel_i)+
                 'K @ z ='+str(round(redshift, 2))+' & T_exp='+str(exptime)+' sec',
                 color=cmap(i_T_tel / max(1, float(len(T_tel)-1))))
    ax1.set_xlim(1, 5)
    #ax1.set_ylim(0.5*min(SNR_line[i_T_tel, :]), max(2.*SNR_line[i_T_tel, :]))
    ymin = max(np.mean(SNR_line[i_T_tel, :])-3.*np.std(SNR_line[i_T_tel, :]),0)
    ymax = np.mean(SNR_line[i_T_tel, :])+3.*np.std(SNR_line[i_T_tel, :])
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlabel('Observed Wavelength [$\mu m$]', fontsize=20)
    ax1.set_ylabel('SNR$_{line}$', fontsize=20)
    ax1.legend(loc=0, fontsize=8)
    #ax1.set_yscale('log')
    ax4 = ax1.twiny()
    ax4.set_xlim(1./(1+redshift), 5./(1+redshift))
    ax4.set_xlabel('Rest-frame Wavelength [$\mu m$]', fontsize=20)

    for i_T_tel, T_tel_i in enumerate(T_tel):
        ax2.plot(wave*1e6, S_line_SNR[i_T_tel, :],
                 label='S_line for SNR_line = '+str(SNR_flambda[0])+
                 ' & Background @ '+str(T_tel_i)+'K & T_exp='+str(exptime)+' sec',
                 color=cmap(i_T_tel / max(1, float(len(T_tel)-1))))
    ax2.set_xlim(1, 5)
    #ax2.set_ylim(0.5*min(S_line_SNR[i_T_tel, :]), max(2.0*S_line_SNR[i_T_tel, :]))
    ymin = max(np.mean(S_line_SNR[i_T_tel, :])-3.*np.std(S_line_SNR[i_T_tel, :]),0)
    ymax = np.mean(S_line_SNR[i_T_tel, :])+3.*np.std(S_line_SNR[i_T_tel, :])
    ax2.set_ylim(ymin, ymax)
    ax2.set_xlabel('Observed Wavelength [$\mu m$]', fontsize=20)
    ax2.set_ylabel('S$_{line}$ [$erg/cm^2/s$]', fontsize=20)
    #ax2.set_yscale('log')
    ax2.legend(loc=0, fontsize=8)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    if create_table == True:

        filename=OUT_DIR+str(idx)+'_spectro_line.txt'
        line = open(filename, 'w')
        line.write('# Wave SNR_line S_line_SNR T_tel \n')

        for i_T_tel, T_tel_i in enumerate(T_tel):
            for iwave, wavelength in enumerate(wave):
                line.write('%.5f %2s' %(wavelength*1e6, '  '))
                line.write('%.2f %2s' %(SNR_line[i_T_tel, iwave], '  '))
                line.write('%.2e %2s' %(S_line_SNR[i_T_tel, iwave], '  '))
                line.write('%.1f %2s' %(T_tel_i, '  '))
                line.write('\n')

        line.close()

    fig4.savefig(OUT_DIR + str(idx) + '_line.pdf')
    plt.close(fig4)

    #plt.show()
#----------------------------------------------------------------------
def PlotSimSpectrum(wave, f_nu_nonoise, f_nu_noise, S_line, T_tel,
                      redshift, logM_star, logsfr10Myrs, A_FUV, logL_dust,
                      FLARE_filters, FLARE_lambda, FLARE_fluxes, FLARE_SNR,
                      exptime, idx, create_table):

    """
    Plots photometric noisy specturm from input CIGALE spectrum and exposure time.

    Inputs:
    - wave [m]
    - SNR []
    - f_nu_nonoise []
    - f_nu_noise []
    - T_tel [K]

    """
    # Directory where the output files are stored
    OUT_DIR = "out/sim/"


    fig5 = plt.figure(figsize=(16, 12))
    fig5.canvas.set_window_title('Spectrum rest-frame w/ and redshifted w/o noise')
    ax1 = fig5.add_subplot(211)
    ax2 = fig5.add_subplot(212)
    #ax3 = fig5.add_subplot(413)
    #ax4 = fig5.add_subplot(414)

    ax1.plot(wave*1e6, f_nu_nonoise,
                 label='Initial modelled spectrum for T_exp='+str(exptime)+' sec',
                 color='k')
    ax1.set_xlim(1., 5.)
    #ax1.set_ylim(0.5*min(f_nu_nonoise), max(2.*SNR_line))
    #ymin = max(np.mean(f_nu_nonoise)-3.*np.std(f_nu_nonoise),0)
    ymax = np.mean(f_nu_nonoise)+5.*np.std(f_nu_nonoise)
    ymin = -0.20*ymax
    #ymax = np.max(f_nu_nonoise)
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlabel('Observed Wavelength [$\mu m$]', fontsize=20)
    ax1.set_ylabel('S$_{lambda}$ [$erg/cm^2/s/A$]', fontsize=20)
    leg1 = ax1.legend(loc=0, fontsize=12)
    leg1.get_frame().set_alpha(0.5)
    #ax1.set_yscale('log')
    ax1.grid(True)
    ax4 = ax1.twiny()
    ax4.set_xlim(1./(1+redshift), 5./(1+redshift))
    ax4.set_xlabel('Rest-frame Wavelength [$\mu m$]', fontsize=20)

    from astropy.convolution import convolve, Box1DKernel
    cmap = mpl.cm.rainbow_r
    for i_T_tel, T_tel_i in enumerate(T_tel):
        #ax2.plot(wave*1e6, f_nu_noise[i_T_tel],
        #         label='Simulated modelled spectrum @ T_tel = '+str(T_tel_i)+
        #         'K & z = '+str(round(redshift,2))+' & T_exp = '+str(exptime)+' sec',
        #         color=cmap(i_T_tel / max(1, float(len(T_tel)-1))),
        #         alpha = 0.3)
        #ax2.plot(wave*1e6, convolve(f_nu_noise[i_T_tel], Box1DKernel(2)),
        #         label='Smoothed simulated modelled spectrum @ T_tel = '+str(T_tel_i)+'K',
        #         color='k', alpha = 1.0)
        ax2.plot(wave*1e6, convolve(f_nu_noise[i_T_tel], Box1DKernel(2)),
                 label='Simulated modelled spectrum @ T_tel = '+str(T_tel_i)+'K' +
                 ' & z = '+str(round(redshift,2))+' & T_exp = '+str(exptime)+' sec' +
                 ' \n logM_star[M_Sun] = '+str(round(logM_star,2)) +
                 ' & logSFR10Myrs[M_Sun/yr] = '+str(round(logsfr10Myrs, 2)) +
                 '\n A_FUV = '+str(round(A_FUV, 3)) +
                 ' & logL_dust[L_Sun] = '+str(round(logL_dust, 2)),
                 color=cmap(i_T_tel / max(1, float(len(T_tel)-1))))
        ax2.plot(FLARE_lambda*1e6, FLARE_fluxes*(3e18/(1e10*FLARE_lambda)**2),
                 label='FLARE Broad-bands', marker='s', markersize=12,
                 linestyle='None', color='k')
        for i_band in range(len(FLARE_lambda)):
            if FLARE_fluxes[i_band] > 1e-38:
                ax2.annotate(FLARE_filters[i_band]+'='+
                         str(round(-2.5*np.log10(FLARE_fluxes[i_band])-48.56, 1)),
                     xy=(0.05+FLARE_lambda[i_band]*1e6,
                         FLARE_fluxes[i_band]*(3e18/(1e10*FLARE_lambda[i_band])**2)),
                     rotation=0, size=8)

    ax2.set_xlim(1, 5)

    #ax2.set_ylim(0.5*min(f_nu_noise), max(2.0*f_nu_noise))
    #ymin = max(np.mean(f_nu_noise[i_T_tel])-3.*np.std(f_nu_noise[i_T_tel]),0)
    ymax = np.mean(f_nu_noise[i_T_tel])+5.*np.std(f_nu_noise[i_T_tel])
    #ymin = -0.20*ymax
    #ymax = np.max(f_nu_noise[i_T_tel])
    ax2.set_ylim(ymin, ymax)
    ax2.set_xlabel('Observed Wavelength [$\mu m$]', fontsize=20)
    ax2.set_ylabel('S$_{lambda}$ [$erg/cm^2/s/A$]', fontsize=20)
    #ax2.set_yscale('log')
    leg2 = ax2.legend(loc=0, fontsize=12, numpoints=1)
    leg2.get_frame().set_alpha(0.5)

    ax2.grid(True)

    #ax1.set_ylim(ax2.get_ylim())
    x_annotate = [0.1216, 0.1240, 0.1549, 0.1640 , 0.1665, 0.1909, 0.3726, 0.3729, 0.4101,
                  0.4340, 0.4959, 0.4861, 0.5007, 0.6548, 0.6563, 0.6584, 0.6716, 0.6730,
                  1.0941, 1.2821, 1.644, 1.8756]
    text_annotate=['Lyalpha',' NV1240', 'CIV1549', 'HeII1640', 'OIII]1665', 'CIII]1909',
                   '[OII]3726', '[OII]3729', 'Hdelta4101', 'Hgamma4340', '[OIII]4959',
                   'Hbeta4861', '[OIII]5007', '[NII]6548', 'Halpha6563', '[NII]6584',
                   '[SII]6716', '[SII]6730', 'Pgamma1.0941', 'Pbeta1.2821', '[FeII]1.644',
                   'Palpha1.8756']
    y_annotate = 0.9*ax1.get_ylim()[1]

    for i_line in range(len(text_annotate)):
        ax1.axvline(x_annotate[i_line]*(1.+redshift), color='k',
                    linestyle='--', linewidth=0.5)
        ax1.annotate(text_annotate[i_line],
                     xy=(x_annotate[i_line]*(1.+redshift), y_annotate),
                     rotation=90, size=8)
        ax2.axvline(x_annotate[i_line]*(1.+redshift), color='k',
                     linestyle='--', linewidth=0.5, alpha=0.3)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    if create_table == True:

        filename=OUT_DIR+str(idx)+'_simu_spectrum.txt'
        simulation = open(filename, 'w')
        simulation.write('# Wave model noisy_model smoothed_noisy_model T_tel \n')

        for i_T_tel, T_tel_i in enumerate(T_tel):
            f_nu_smoothed = convolve(f_nu_noise[i_T_tel, :], Box1DKernel(2))
            for iwave, wavelength in enumerate(wave):
                simulation.write('%.5f %2s' %(wavelength*1e6, '  '))
                simulation.write('%.2e %2s' %(f_nu_nonoise[iwave], '  '))
                simulation.write('%.2e %2s' %(f_nu_noise[i_T_tel, iwave], '  '))
                simulation.write('%.2e %2s' %(f_nu_smoothed[iwave], '  '))
                simulation.write('%.1f %2s' %(T_tel_i, '  '))
                simulation.write('\n')

        simulation.close()

    fig5.savefig(OUT_DIR + str(idx) + '_sim.pdf')
    plt.close(fig5)

    #plt.show()
#
#----------------------------------------------------------------------
def PlotSFH(time_grid, sfr, redshift, logM_star, logsfr10Myrs,
             A_FUV, logL_dust, idx, create_table):
    """
    This saves the Star Formation History in a file.
    """
#
    # Directory where the output files are stored
    OUT_DIR = "out/sfh/"

    fig6 = plt.figure(figsize=(16, 10))
    fig6.canvas.set_window_title('Star Formation History')
    ax1 = fig6.add_subplot(111)

    ax1.plot(time_grid, sfr, 'k', label='Star Formation History[M_Sun/yr]' +
                                ' to z = '+str(round(redshift, 2))+' => \n' +
                                ' logM_star[M_Sun] = '+str(round(logM_star,2))+
                                ' & logSFR10Myrs[M_Sun/yr] = '+str(round(logsfr10Myrs, 2)) +
                                '\n & A_FUV = '+str(round(A_FUV, 3)) +
                                ' & logL_dust[L_Sun] = '+str(round(logL_dust, 2)))

    ax1.set_xlabel('Time [Myr]', fontsize=20)
    ax1.set_ylabel('SFH [M$_\odot$ yr$^{-1}$]', fontsize=20)
    ax1.legend(loc=0, fontsize=12)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    if create_table == True:

        filename=OUT_DIR+str(idx)+'_sfh.txt'
        sfh = open(filename, 'w')
        sfh.write('# Time SFR \n')

        for itime, time in enumerate(time_grid):
                sfh.write('%.5f %2s' %(time_grid[itime], '  '))
                sfh.write('%.2e %2s' %(sfr[itime], '  '))
                sfh.write('\n')

        sfh.close()

    fig6.savefig(OUT_DIR + str(idx) + '_sfh.pdf')
    plt.close(fig6)

    #plt.show()


#----------------------------------------------------------------------
def SolveQuadratic(a, b, c):
    """
    To estimate the fluxes, we solve a quadratic equation.

    SNR = Signal / sqrt (Signal + Background + Dark + Readout)
    SNR**2 = Signal**2 / (Signal + Background + Dark + Readout)
    SNR**2 * (Signal + Background + Dark + Readout) = Signal**2
    - Signal**2  +  SNR**2 * Signal  +  SNR**2 * (Background + Dark + Readout) = 0
    Signal**2 - SNR**2 * Signal - SNR**2 * (Background + Dark + Readout) = 0

    which is a quadratic equation: ax**2 + b*x + c = 0

    and:
    Delta > 0 => solution is (-b + sqtr(Delta)) / (2a)
    """
    Delta = b**2 - (4 * a * c)
    x = -1. * b + np.sqrt(Delta) / (2 * a)

    return x

#
#----------------------------------------------------------------------
#
def init_simulation(params, filters, save_sfh, create_table, flag_background, flag_phot,
                    flag_spec, flag_line, flag_sim,lambda_norm, mag_norm, exptime, SNR,
                    S_line, fluxes, info, t_begin, n_computed):
    """Initializer of the pool of processes. It is mostly used to convert
    RawArrays into numpy arrays. The latter are defined as global variables to
    be accessible from the workers.

    Parameters
    ----------
    params: ParametersHandler
        Handles the parameters from a 1D index.
    filters: List
        Contains the names of the filters to compute the fluxes.
    save_sfh: boolean
        Indicates whether the SFH should be saved.
    fluxes: RawArray and tuple containing the shape
        Fluxes of individual models. Shared among workers.
    n_computed: Value
        Number of computed models. Shared among workers.
    t_begin: float
        Time of the beginning of the computation.
,
    """
    global gbl_model_fluxes, gbl_model_info, gbl_n_computed, gbl_t_begin
    global gbl_params, gbl_previous_idx, gbl_filters, gbl_save_sfh
    global gbl_warehouse, gbl_keys
    global gbl_lambda_norm, gbl_mag_norm, gbl_exptime, gbl_SNR, gbl_S_line
    global gbl_create_table, gbl_flag_background, gbl_flag_phot
    global gbl_flag_spec, gbl_flag_line, gbl_flag_sim

    gbl_model_fluxes = np.ctypeslib.as_array(fluxes[0])
    gbl_model_fluxes = gbl_model_fluxes.reshape(fluxes[1])

    gbl_model_info = np.ctypeslib.as_array(info[0])
    gbl_model_info = gbl_model_info.reshape(info[1])

    gbl_n_computed = n_computed
    gbl_t_begin = t_begin

    gbl_params = params

    gbl_previous_idx = -1

    gbl_filters = filters

    gbl_save_sfh = save_sfh

    gbl_lambda_norm = lambda_norm
    gbl_mag_norm = mag_norm
    gbl_exptime = exptime
    gbl_SNR = SNR
    gbl_S_line = S_line

    gbl_create_table = create_table
    gbl_flag_background = flag_background
    gbl_flag_phot = flag_phot
    gbl_flag_spec = flag_spec
    gbl_flag_line = flag_line
    gbl_flag_sim = flag_sim

    gbl_warehouse = SedWarehouse()

    gbl_keys = None

def simulation(idx):

    """Simulates observations.

    Parameters
    ----------
    sed: pcigale.sed.SED object

    """

    global gbl_previous_idx, gbl_keys
    global gbl_lambda_norm, gbl_mag_norm, gbl_exptime, gbl_SNR, gbl_S_line
    global gbl_create_table, gbl_flag_background, gbl_flag_phot
    global gbl_flag_spec, gbl_flag_line, gbl_flag_sim, gbl_save_sfh

    lambda_norm = gbl_lambda_norm
    mag_norm = gbl_mag_norm
    exptime = gbl_exptime
    SNR = gbl_SNR
    S_line = gbl_S_line

    save_sfh = gbl_save_sfh
    create_table = gbl_create_table
    flag_background = gbl_flag_background
    flag_phot = gbl_flag_phot
    flag_spec = gbl_flag_spec
    flag_line = gbl_flag_line
    flag_sim = gbl_flag_sim

    if gbl_previous_idx > -1:
        gbl_warehouse.partial_clear_cache(
            gbl_params.index_module_changed(gbl_previous_idx, idx))
    gbl_previous_idx = idx

    sed = gbl_warehouse.get_sed(gbl_params.modules,
                                gbl_params.from_index(idx))

    # Which project?
    Project  = 'FLARE' # or 'JWST'

    # Directory where the output files are stored
    OUT_DIR = "out/"
    redshift = sed.info['universe.redshift']
    distance = sed.info['universe.luminosity_distance'] # in metres

    # Physical and mathematical constants
    h = 6.626e-34 # J.s or J/Hz
    c = 3.0e8    # m/s
    k = 1.38e-23  # J/K
    pi = 3.1415926

    # Retrieve the final computed SED using all the previous modules
    # including the IGM and the redshifting. In other words,
    # this module must be the last one. Note that it does require
    # an SFH and an SSP module but nothing else (except redshifting)

    # Wavelengths are in nanometers and observed.
    wavelength = sed.wavelength_grid # observed nanometers
    # Convert nm to Hz, still observed-frame.
    nu = lambda_to_nu(wavelength) # observed Hz

    # We select the spectrum in the observed frame from 1 to 5 um for models
    w_z = np.where((wavelength >= 1000.) & (wavelength <= 5000.))

    # Retrieve the parameters necessary to define the telescope FLARE
    onexposure_ima, onexposure_spec, pixel_ima, pixel_spec, slice, \
        eta_tel, EE2um, Omega_ima, A, R_ima, R_spec, Omega_spec, eta_spec, \
        T_tel_min, T_tel_max, Delta_T_tel, emissivity = \
        set_config(Project, wavelength[w_z])
    RON, Dark, QE, ADU = Detector(Project, wavelength[w_z])

    # Luminosity is in W/nm. Note that this is not true luminosity,
    # see redshifting.py where we redshift the SED wavelength grid with
    # "*(1+redshift)" and we modify the luminosity contribution to keep energy
    # constant with "(1. + redshift)".
    luminosity = sed.luminosity

    # We convert from W/nm -> W/m -> W/Hz -> W/Hz/cm2 -> erg/s/Hz/cm2 (obs-frame)
    f_nu = luminosity[w_z] * 1e9 * (1e-9*wavelength[w_z])**2/c / \
           (4*pi*(1e2*distance)**2) * 1e7

    if (lambda_norm == '' or mag_norm == ''):
        print('')
        #print('Nothing special to do: we directly use the output from CIGALE')
    else:
        w_norm = np.where(wavelength[w_z] == lambda_norm)
        if not w_norm:
            sys.exit('Normalisation wavelength outside FLARE\'s 1 - 5 um range')

    #----------------------------------------------------------------------
    #
    # If requested, we save the SFH in a file.
    time_grid = sed.sfh[0]
    sfr = sed.sfh[1]

    # BC03 SSPS used
    if 'stellar.m_star' in sed.info:
        M_star = sed.info['stellar.m_star']
        M_star_old = sed.info['stellar.m_star_old']
        M_star_young = sed.info['stellar.m_star_young']

    # Maraston (2005) SSPs used
    elif 'stellar.mass_total' in sed.info:
        M_star = sed.info['stellar.mass_total']
        M_star_old = sed.info['stellar.mass_total_old']
        M_star_young = sed.info['stellar.mass_total_young']

    sfh_integrated = sed.info['sfh.integrated']
    sfr_inst = sed.info['sfh.sfr']
    sfr10Myrs = sed.info['sfh.sfr10Myrs']
    sfr100Myrs = sed.info['sfh.sfr100Myrs']

    A_FUV = sed.info['attenuation.FUV']
    L_Sun = 3.846e26 # Watt
    logL_dust = np.log10(sed.info['dust.luminosity'] / L_Sun)

    PlotSFH(time_grid, sfr, redshift, np.log10(M_star), np.log10(sfr10Myrs),
             A_FUV, logL_dust, idx, create_table)
    #----------------------------------------------------------------------
    #
    #----------------------------------------------------------------------
    #
    # Create the Zodiacal light component
    Zodi = ZodiacalLight(1e-9*wavelength[w_z]) # in erg/s/cm2/A/arcsec2
    #----------------------------------------------------------------------
    #
    # Create the thermal background from the telescope.
    # A range of temperatures can be provided.
    N_T_tel = (T_tel_max - T_tel_min) / Delta_T_tel + 1
    T_tel = np.linspace(T_tel_min, T_tel_max, N_T_tel)
    Thermal_from_telescope = np.zeros((len(T_tel), len(wavelength[w_z]), ))
    Background = np.zeros((len(T_tel), len(wavelength[w_z]), ))

    for i_T_tel, T_tel_i in enumerate(T_tel):
        # Thermal_from_telescope in units of erg/s/cm2/A/arcsec2
        Thermal_from_telescope[i_T_tel, :] = \
                         emissivity * blackbody_lam(1e-9*wavelength[w_z], T_tel_i)
    #----------------------------------------------------------------------
    #
    # Sum up the different components of the background in erg/s/cm2/A/arcsec2

        Background[i_T_tel, :] = Zodi + Thermal_from_telescope[i_T_tel, :]

    if flag_background == True:
        PlotBackground(Project, 1e-9*wavelength[w_z], Zodi,
                                Thermal_from_telescope, T_tel, redshift, idx, create_table)

    # ----------------------------------------------------------------------------
    #       Now Photometric ETC: compute SNR for the modelled spectrum
    #
    # We start by computing the flux densities in the FLARE filters
    # Note that the filters need to be in the CIGALE database.
    Lyman_alpha = 121.6*1e-9

    # We read the transmissions of the filters
    FLARE_filters = np.array(['F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W'])
    FLARE_lambda = np.array([1.15e-6, 1.50e-6, 2.00e-6, 2.77e-6, 3.56e-6, 4.44e-6]) # in m
    FLARE_SNR = np.zeros((len(T_tel), len(FLARE_lambda)))
    S_nu_SNR = np.zeros((len(T_tel), len(FLARE_lambda)))
    lambda_min = np.zeros_like(FLARE_lambda)
    lambda_max = np.zeros_like(FLARE_lambda)
    lambda_eff = np.zeros_like(FLARE_lambda)

    # We will resample the wavelength and properties ('_r' arrays) to get N_wave_r values
    N_wave_r = len(f_nu)
    transmission_r = np.zeros((len(FLARE_lambda), N_wave_r))
    SNR_phot = np.zeros((len(T_tel), N_wave_r))

    for ifilt, filter_ in enumerate(FLARE_filters):
        with Database() as db:
                filter_ = db.get_filter(filter_)
        lambda_eff[ifilt] = filter_.effective_wavelength # in observed nm
        lambda_min[ifilt] = filter_.trans_table[0][0] # in observed nm
        lambda_max[ifilt] = filter_.trans_table[0][-1] # in observed nm

        # Observed wavelength in nm and observed frequencies in Hz
        wavelength_r = np.linspace (lambda_min[ifilt], lambda_max[ifilt], N_wave_r)
        nu_r = lambda_to_nu(wavelength_r)

        # Initialize the detector characteristics.
        RON_r, Dark_r, QE_r, ADU_r = Detector(Project, wavelength_r)

        # Evaluate the function that computes the re-sampled filter transmission
        filt = InterpolatedUnivariateSpline(filter_.trans_table[0],
                                            filter_.trans_table[1], k=1)
        # We regrid both spectrum and filter to the best wavelength grid
        # to avoid interpolating a high wavelength density curve to a low
        # density one. Also, we limit the work wavelength domain to the
        # filter one.

        # Resampling of the normalized (max = 1.0) transmission of each FLARE filter
        transmission_r[ifilt,:] = filt(wavelength_r) / np.max(filt(wavelength_r))

        # Resampling of the flux density to the same grid ('_r')
        #print('<<', wavelength[w_z], f_nu)
        f_nu_r = np.interp(wavelength_r, wavelength[w_z], f_nu)

        for i_T_tel, T_tel_i in enumerate(T_tel):

            # Resampling of the background to the same grid ('_r')
            Background_r = np.interp(wavelength_r, wavelength[w_z], Background[i_T_tel, :])

            # flux_c = source count rate [ph/s] and f_nu [erg/s/cm2/Hz]
            flux_c = f_nu_r * transmission_r[ifilt,:] * EE2um * A * eta_tel * QE_r * \
                            (nu_r/R_ima) / 1e7 / (h*nu_r)

            # N = noise, with N = [S_c + BG_c + RON^2]^{1/2}  where "_c" means counts
            # in ph/s with BG = background counts from sky, dark, and thermal
            # (telescope and instrument) and RON = read-out noise.

            # Signal in ph
            Signal_0 = exptime * flux_c

            # Background in erg/s/cm2/A/arcsec2 -> ph
            Background_0 = Background_r * eta_tel * A * Omega_ima * pixel_ima**2 * \
                       QE_r / 1e7 / (h*nu_r) * 10.*(wavelength_r/R_ima) * exptime
            Dark_0 = Dark_r / ADU_r * Omega_ima * exptime
            Readout_0 = (exptime/onexposure_ima)*RON_r**2 * Omega_ima
            SNR_phot[i_T_tel, :] = Signal_0 / \
                               np.sqrt(Signal_0 + Background_0 + Dark_0 + Readout_0)

            # We compute the average of SNR over the wavelength range
            FLARE_SNR[i_T_tel, ifilt] = \
                 np.mean(SNR_phot[i_T_tel, :]*transmission_r[ifilt,:])

        # Compute mAB/Snu for a given SNR
        # From http://home.strw.leidenuniv.nl/~brandl/OBSTECH/Handouts_24Sep.pdf

            SNR_mAB = SNR * np.ones_like(wavelength_r) # limiting SNR
        # S_c_SNR in ph/s
        # Signal**2 - SNR**2 * Signal - SNR**2 * (Background + Dark + Readout) = 0
            S_c_SNR = SolveQuadratic(1.,
                                 -1.*SNR_mAB**2,
                                 -1.*SNR_mAB**2 * (Background_0 + Dark_0 + Readout_0))

            S_nu_SNR_0 = S_c_SNR / transmission_r[ifilt,:] / (EE2um * A * eta_tel * QE_r * \
                             (nu_r/R_ima) / 1e7 / (h*nu_r)) / exptime # erg/s/cm2/Hz

            S_nu_SNR[i_T_tel, ifilt] = \
                           np.mean(S_nu_SNR_0*transmission_r[ifilt,:])

    FLARE_fluxes = np.array([sed.compute_fnu(filter_) * 1e-26 for
                                filter_ in FLARE_filters]) # in erg/cm2/s/Hz
    FLARE_fluxes[np.where(FLARE_fluxes < 1e-38)] = 1e-38
    FLARE_mAB = -2.5* np.log10(FLARE_fluxes) - 48.56

    # 1 Jy = 10**-23 erg/s/cm2/Hz =>  1 nJy = 10**-32 erg/s/cm2/Hz
    S_nu_SNR = S_nu_SNR * 1e32 # in nJy
    S_nu_mAB = -2.5* np.log10(FLARE_fluxes) + 16.4

    if flag_phot == True:
        PlotPhot(S_nu_mAB, S_nu_SNR, SNR_mAB[0], lambda_min, lambda_max, lambda_eff,
             FLARE_filters, FLARE_fluxes, FLARE_mAB, FLARE_SNR,
             T_tel, redshift, exptime, R_ima, idx, create_table)

# ----------------------------------------------------------------------------------------
#       Now Spectroscopic ETC for continuum
#
    #print('------------ SNR_spec for modelled spectrum')

    # Nebular emission to be subtracted here because we're interested in continuum
    nebular_lines_young = sed.get_lumin_contribution('nebular.lines_young')[w_z] * \
                          1e9 * (1e-9*wavelength[w_z])**2/c / \
                          (4*pi*(1e2*distance)**2) * 1e7
    nebular_lines_old = sed.get_lumin_contribution('nebular.lines_old')[w_z] * \
                          1e9 * (1e-9*wavelength[w_z])**2/c / \
                          (4*pi*(1e2*distance)**2) * 1e7
    att_nebular_lines_young = \
                   sed.get_lumin_contribution('attenuation.nebular.lines_young')[w_z] * \
                   1e9 * (1e-9*wavelength[w_z])**2/c / \
                   (4*pi*(1e2*distance)**2) * 1e7
    att_nebular_lines_old = \
                   sed.get_lumin_contribution('attenuation.nebular.lines_old')[w_z] * \
                   1e9 * (1e-9*wavelength[w_z])**2/c / \
                   (4*pi*(1e2*distance)**2) * 1e7

    flux_cont = (f_nu - nebular_lines_young - nebular_lines_old - \
                      + att_nebular_lines_young + att_nebular_lines_old) * \
                        EE2um * A * eta_tel * eta_spec * QE * (nu[w_z]/R_spec) / \
                        1e7 / (h*nu[w_z]) # ph/s
    # We need to bin to get the correct spectral resolution Delta_lambda = lambda/R
    # Because Delta_lambda for gratings is about constant, R depends on lambda
    # We want to get 1024 pixels (2 per resolution element in the range 1.25 - 2.5 um)
    # Delta_lambda = (2500 - 1250) / 1024 = 1.22 nm => 2.44 nm per pixel for octave #1
    # We want to get 1024 pixels (2 per resolution element in the range 2.5 - 5.0 um)
    # Delta_lambda = (5000 - 2500) / 1024 = 2.44 nm => 4.88 nm per pixel for octave #2

    # However, with these models, we have less data than needed => interpolation

    wave_octave1 = np.linspace(1250., 2500., 1024) # in nm
    wave_octave2 = np.linspace(2500., 5000., 1024) # in nm
    new_wavelength = np.hstack((wave_octave1, wave_octave2)) # in nm
    new_frequency = c / (1e-9*new_wavelength) # in Hz

    # We define masks for the two octaves
    w_octave1 = np.where((wavelength[w_z] >= 1250.) & (wavelength[w_z] <= 2500.))
    w_octave2 = np.where((wavelength[w_z] >= 2400.) & (wavelength[w_z] <= 5000.))

    # We interpolate the flux
    f1 = InterpolatedUnivariateSpline(wavelength[w_z][w_octave1],
                                      flux_cont[w_octave1], k=1)
    f2 = InterpolatedUnivariateSpline(wavelength[w_z][w_octave2],
                                      flux_cont[w_octave2], k=1)
    #
    spec_octave1 = f1(wave_octave1)
    spec_octave2 = f2(wave_octave2)
    new_spec = np.hstack((spec_octave1, spec_octave2))

    # We re-create the background with the new wavelength grid
    #----------------------------------------------------------------------
    #
    # Create the Zodiacal light component
    Zodi = ZodiacalLight(1e-9*new_wavelength) # in erg/s/cm2/A/arcsec2

    #----------------------------------------------------------------------
    #
    # Create the thermal background from the telescope.
    # A range of temperatures can be provided.

    Thermal_from_telescope = np.empty((len(T_tel),len(new_wavelength), ))
    Thermal_from_telescope[:] = np.NAN

    for i_T_tel, T_tel_i in enumerate(T_tel):
        # Thermal_from_telescope in units of erg/s/cm2/A/arcsec2
        Thermal_from_telescope [i_T_tel, :] = \
                             emissivity * blackbody_lam(1e-9*new_wavelength, T_tel_i)

    #----------------------------------------------------------------------
    #
    # Sum up the different components of the background in erg/s/cm2/A/arcsec2
    Background = Zodi + Thermal_from_telescope

    # Read the detector's characteristics.
    RON, Dark, QE, ADU = Detector(Project, new_wavelength)

    Signal_1 = new_spec * exptime
    Background_1 = Background * eta_tel * eta_spec * A * Omega_ima * \
                     pixel_spec**2 * QE / 1e7 / (h*new_frequency) * \
                     10.*(new_wavelength/R_ima) * exptime
    Dark_1 = Dark / ADU * Omega_spec * exptime
    Readout_1 = (exptime/onexposure_spec)*RON**2 * Omega_spec

    SNR_spec = Signal_1 / np.sqrt(Signal_1 + Background_1 + Dark_1 + Readout_1)

    # Compute mAB/Sp_nu for a given SNR
    # From http://home.strw.leidenuniv.nl/~brandl/OBSTECH/Handouts_24Sep.pdf

    #print('------------ S_nu_spec for SNR_spec=Cte')
    SNR_mAB = SNR * np.ones_like(new_wavelength) # limiting SNR

    S_c_SNR = SolveQuadratic(1.,
                               -1.*SNR_mAB**2,
                               -1.*SNR_mAB**2 * (Background_1 + Dark_1 + Readout_1))

    S_nu_SNR = S_c_SNR / (EE2um * A * eta_tel * eta_spec * QE * \
                            (new_frequency/R_spec) / \
                            1e7 / (h*new_frequency)) / exptime # erg/s/cm2/Hz
    mAB_SNR = -2.5*np.log10(S_nu_SNR) - 48.6 # AB magnitude
    # 1 Jy = 10**-23 erg/s/cm2/Hz =>  1 nJy = 10**-32 erg/s/cm2/Hz
    S_nu_SNR = S_nu_SNR * 1e32 # in nJy

    if flag_spec == True:
         PlotSpec(1e-9*new_wavelength, SNR_spec, SNR_mAB, mAB_SNR, S_nu_SNR,
                  T_tel, redshift, exptime, idx, create_table)

# ----------------------------------------------------------------------------------------
#       Now Spectroscopic ETC for line spectrum
#
    #print('------------ SNR_line from model')

    #S_c = source count rate and S_nu [erg/s/cm2] = 10^(-(mAB+48.6)/2.5)
    flux_c = S_line * EE2um * A * eta_tel * eta_spec * QE / 1e7 / (h*new_frequency) # ph/s

    Signal_2 = flux_c * exptime
    Background_2 = Background * eta_tel * eta_spec * A * Omega_spec * \
                   pixel_spec**2 * QE / 1e7 / (h*new_frequency) * \
                   10.*(new_wavelength/R_spec) * exptime

    Dark_2 = Dark / ADU * Omega_spec * exptime
    Readout_2 = (exptime/onexposure_spec)*RON**2 * Omega_spec

    SNR_line = Signal_2 / np.sqrt(Signal_2 + Background_2 + Dark_2 + Readout_2)

    # Compute mAB/Sp_nu for a given SNR
    # From http://home.strw.leidenuniv.nl/~brandl/OBSTECH/Handouts_24Sep.pdf

    SNR_flambda = SNR * np.ones_like(new_wavelength) # limiting SNR
    err_flambda = 1.00 / SNR_flambda

    #print('------------ S_lambda_line for SNR_line=Cte')

    S_c_SNR = SolveQuadratic(1.,
                             -1.*SNR_flambda**2,
                             -1.*SNR_flambda**2 * (Background_2 + Dark_2 + Readout_2))

    S_line_SNR = S_c_SNR / (EE2um * A * eta_tel * eta_spec * QE / 1e7 / \
                (h*new_frequency / exptime)) # erg/s/cm2

    if flag_line:
        PlotLine(1e-9*new_wavelength, S_line, SNR_line, SNR_flambda, mAB_SNR,
                 S_line_SNR, T_tel, redshift, exptime, idx, create_table)

# ----------------------------------------------------------------------------------------
#       Now we simulate the observation of a noisy spectrum
#

    # From http://www.cfht.hawaii.edu/Instruments/Imaging/WIRCam/dietWIRCam.html
    # 2.7 Relation between SNR and photometry quality, we find:
    # SNR = 3 : Detection - Flux error = 33% - mag error = 0.31
    # SNR = 7 : Fair detection - Flux error = 15% - mag error = 0.13
    # SNR = 15 : Good detection - Flux error = 7% - mag error = 0.06
    # SNR = 25 : Quality photometry - Flux error = 4% - mag error = 0.04
    # SNR = 100 : High quality photometry - Flux error = 1% - mag error = 0.009
    # We assume Flux error = 100 / SNR (%)

    f1 = InterpolatedUnivariateSpline(wavelength[w_z][w_octave1],
                                      f_nu[w_octave1], k=1)
    f2 = InterpolatedUnivariateSpline(wavelength[w_z][w_octave2],
                                      f_nu[w_octave2], k=1)
    #
    f_nu_octave1 = f1(wave_octave1)
    f_nu_octave2 = f2(wave_octave2)
    new_f_nu = np.hstack((f_nu_octave1, f_nu_octave2)) # in erg/cm2/s/Hz
    new_f_lambda = new_f_nu * (1.e10*c/((10.*new_wavelength)**2)) # in erg/cm2/s/A

    err_f_lambda = np.empty_like(SNR_line)
    noisy_f_lambda = np.empty_like(SNR_line)
    for i_T_tel, T_tel_i in enumerate(T_tel):
        err_f_lambda[i_T_tel] = 1.00 / SNR_line[i_T_tel]

    for i_lam in range(len(new_f_lambda)):
        noisy_f_lambda[i_T_tel] = np.random.normal(
                new_f_lambda+S_line_SNR[i_T_tel,:]/(10.*(new_wavelength/R_spec)),
                (new_f_lambda+S_line_SNR[i_T_tel,:]/(10.*(new_wavelength/R_spec)))*
                err_f_lambda[i_T_tel]
                                                   )

    if flag_sim == True:
        PlotSimSpectrum(1e-9*new_wavelength, new_f_lambda, noisy_f_lambda,
                          S_line_SNR / (10.*(new_wavelength/R_spec)), T_tel, redshift,
                          np.log10(M_star), np.log10(sfr10Myrs), A_FUV, logL_dust,
                          FLARE_filters, FLARE_lambda, FLARE_fluxes, FLARE_SNR,
                          exptime, idx, create_table)
# ----------------------------------------------------------------------------------------
#       And, finally, we simulate the 3D-observations with FLARE
#
    if flare3d_sim == True:
