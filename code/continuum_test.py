#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import warnings
from tqdm import tqdm
import glob
from urllib import request
import shutil
from tempfile import mkstemp
import time

import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from astropy.io import fits
import astropy.units as u
import astropy.constants as const
from astropy.table import Table, vstack, hstack, join
from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter
from specutils.fitting import find_lines_threshold, find_lines_derivative, fit_lines, fit_generic_continuum
from specutils.manipulation import noise_region_uncertainty, trapezoid_smooth
from specutils import Spectrum1D, SpectralRegion
from astropy.nddata import NDUncertainty
from scipy.optimize import curve_fit


savepath = '/Users/isabelkain/Desktop/GALEX/analysis/continuum'
analysispath = '/Users/isabelkain/Desktop/GALEX/analysis'
datapath = '/Volumes/Seagate/seagate_backup/GALEX/data'

## Helper functions
angstrom2meter = 10**-10
joules2ergs = 10**7


def redshift_correction(wavelengths_observed, z):
    '''
    Finds amount by which stellar emission is redshifted, and subtract this from stellar spectrum.
    Input:
    wavelengths_observed: observed wavelengths of spectrum (Angstroms)
    z: redshift of star
    
    Returns:
    wavelengths_emitted: emitted wavelengths of spectrum (Angstroms)
    '''
    
    wavelengths_emitted = wavelengths_observed / (1 + z)
    return wavelengths_emitted


def blackbody(wavelengths, Teff):
    '''
    Returns blackbody continuum for an object at Teff (K)
    Inputs:
    wavelengths [arr]: (Angstroms)
    Teff [scalar]: effective temperature (K)
    
    Returns:
    B [arr]: blackbody continuum for Teff (unitless)
    '''
    
    wav = wavelengths * angstrom2meter * u.m
    Teff = Teff * u.K
    
    B = (2. * const.h * const.c**2) / (wav**5.) / ( np.exp( (const.h * const.c) / (wav * const.k_B * Teff) ) - 1. )
    
    return (B / B.max()).value


def fitBB_fixedTeff(wavelengths, Teff, a, b):
    '''
    Scale blackbody continuum emission.
    Input:
    wavelengths [arr]: spectrum wavelengths (Angstroms)
    Teff [scalar]: effective temperature of star (K)
    a, b: scaling factors to be optimized by curve_fit
    '''
    return a * blackbody(wavelengths, Teff) + b


def fitBB_fitTeff(wavelengths, Teff, a, b):
    '''
    Fit blackbody emission to spectrum continuum. 
    Input:
    wavelengths [arr]: spectrum wavelengths (Angstroms)
    Teff [scalar]: effective temperature of star (K) (will be optimized by curve_fit)
    a, b: scaling factors to be optimized by curve_fit
    
    Returns:
    -- [arr]: blackbody continuum curve
    '''
    
    wav = wavelengths * angstrom2meter * u.m
    Teff = Teff * u.K
    
    B = (2. * const.h * const.c**2) / (wav**5.) / ( np.exp( (const.h * const.c) / (wav * const.k_B * Teff) ) - 1. )

    return a * (B / B.max()).value + b



if __name__ == "__main__":

    ## Read in table of IUE data
    table = Table.read(f'{analysispath}/dataset.ecsv')


    for i in range(len(table)):
        
        row = table[i]

        ######################
        ## Read in raw data ##
        ######################

        ## Grab data info from table
        swpid = row['obs_id']                 # swp54894
        starname = row['main_id']             # * gam Mic
        objname = starname.replace(' ','')      # *gamMic

        ## Open FITS file
        hdul = fits.open(f'{datapath}/{swpid}.fits')

        ## Record redshift & temperature
        z = row['redshift']
        Teff = row['Fe_H_Teff']

        if type(z) == np.ma.core.MaskedConstant: z = 0.

        ## Always fit Teff for best-fitting blackbody
        if Teff == 0: Teff = 5000. # guess value of 5000 K if no Teff available
        fit_teff = True

        ## Read in wavelength, flux, and fluxerr data

        wavelengths = redshift_correction(hdul[1].data['WAVE'][0], z) # Apply redshift correction
        rawflux = hdul[1].data['FLUX'][0]                             # raw because blackbody continuum will soon be subtracted
        fluxerr = hdul[1].data['SIGMA'][0]

        hdul.close()


        ###########################
        ## BLAKBODY SUBTRACTION  ##
        ###########################

        ## Mask most features to let curve_fit see the underlying emission

        diff = rawflux[1:] - rawflux[:-1]
        diff = np.append(diff, 0.)
        std = np.std(diff)
        mask = np.logical_and((diff <= 2*std), (diff >= -2*std))

        x = wavelengths[mask]
        y = rawflux[mask]
        yerr = fluxerr[mask]


        ## Fit blackbody curve. If Teff known, then fix for curve_fit, else let vary

        if fit_teff:
            popt, pcov = curve_fit(fitBB_fitTeff, x, y, p0=[Teff, 0., 0.],
                                   sigma=yerr, absolute_sigma=True,
                                   bounds=((3000., 0, 0), (9000., y.max(), np.inf))) 

            Teff = popt[0]

        else:
            popt, pcov = curve_fit(lambda x, a, b: fitBB_fixedTeff(x, Teff, a, b), x, y, 
                                   sigma=yerr, absolute_sigma=True,
                                   bounds=((-np.inf, 0), (np.inf, np.inf))) 


        ## Subtract BB curve

        if fit_teff:
            flux = rawflux - fitBB_fitTeff(wavelengths, Teff, popt[1], popt[2])
        else:
            flux = rawflux - fitBB_fixedTeff(wavelengths, Teff, popt[0], popt[1])



        ###########################
        ## CONTINUUM SUBTRACTION ##
        ###########################

        spectrum = Spectrum1D(flux=rawflux*(u.erg/u.cm**2/u.s/u.AA), spectral_axis=wavelengths*u.AA)

        with warnings.catch_warnings():  # Ignore warnings
            warnings.simplefilter('ignore')
            g1_fit = fit_generic_continuum(spectrum)

        y_continuum_fitted = g1_fit(wavelengths*u.AA)



        ###########################
        ## PLOT COMPARISON ##
        ###########################


        fig, ax = plt.subplots(1,2, figsize=(12,5))  

        ## Spectrum with blackbody, continuum plotted
        ax[0].plot(wavelengths, rawflux) 
        ax[0].fill_between(wavelengths, rawflux-fluxerr, rawflux+fluxerr, alpha=0.5) 
        ax[0].plot(wavelengths, fitBB_fitTeff(wavelengths, popt[0], popt[1], popt[2]), 
                   label=f'Blackbody fit:\n{popt[1]:0.2e}func + {popt[2]:0.2e}\nTeff = {popt[0]}')
        ax[0].plot(wavelengths, y_continuum_fitted.value, label='Continuum fit') 

        ## Spectrum with blackbody, continuum subtracted
        ax[1].plot(wavelengths, flux, label='Blackbody subtracted') 
        ax[1].fill_between(wavelengths, flux-fluxerr, flux+fluxerr, alpha=0.5) 

        ax[1].plot(wavelengths, rawflux - y_continuum_fitted.value, label='Continuum subtracted') 
        ax[1].fill_between(wavelengths, (rawflux - y_continuum_fitted.value)-fluxerr, 
                           (rawflux - y_continuum_fitted.value)+fluxerr, alpha=0.5) 

        ax[0].legend()
        ax[1].legend()

        fig.suptitle(f'{swpid} / {starname}', fontsize=16)

        plt.savefig(f'{savepath}/{swpid}.png', bbox_inches='tight')
        plt.close('all')
    

