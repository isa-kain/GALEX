import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import warnings
from tqdm import tqdm
import glob
from urllib import request

import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from astropy.io import fits
import astropy.units as u
import astropy.constants as const
import astropy.coordinates as coord
from astropy.table import Table, vstack, hstack, join
from astropy.modeling import models
from specutils.fitting import find_lines_threshold, find_lines_derivative, fit_lines
from specutils.manipulation import noise_region_uncertainty
from specutils import Spectrum1D, SpectralRegion
from astropy.nddata import NDUncertainty
from scipy.optimize import curve_fit

from astroquery.simbad import Simbad
from astroquery.nist import Nist
from astroquery.xmatch import XMatch
from astroquery.vizier import Vizier
from astroquery.mast import Observations


## Where should analysis results and data be saved to?
global analysispath = '/Users/isabelkain/Desktop/GALEX/analysis'
global datapath = '/Users/isabelkain/Desktop/GALEX/data'


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
    
    wav = wavelengths * aangstrom2meter * u.m
    Teff = Teff * u.K
    
    B = (2. * const.h * const.c**2) / (wav**5.) / ( np.exp( (const.h * const.c) / (wav * const.k_B * Teff) ) - 1. )
    
    return B / B.max()


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
    
    wav = wavelengths * aangstrom2meter * u.m
    Teff = Teff * u.K
    
    B = (2. * const.h * const.c**2) / (wav**5.) / ( np.exp( (const.h * const.c) / (wav * const.k_B * Teff) ) - 1. )

    return a * (B / B.max()) + b


def analyze_spectrum(table):
    '''
    Analyze spectrum given single row of database.
    Input:
    table [astropy Table]: single row of full XMatched dataset
    
    Returns:
    ???
    '''
    
    ## Grab data info from table
    swpid = table['obs_id']                 # swp54894
    starname = table['main_id']             # * gam Mic
    objname = starname.replace(' ','')      # *gamMic
    
    ## Open FITS file
    hdul = fits.open(f'{datapath}/{swpid}.fits')
    
    ## Load file information
    spclass = hdul[1].header['SRCCLASS']    # Object Class 
    camera = hdul[0].header['CAMERA']       # or hdul[1].header['INSTRUME']
    dispersion = hdul[0].header['DISPERSN'] # LOW or HIGH
    fluxcal = hdul[1].header['FLUX_CAL']    # should be ABSOLUTE
    tstart = float(str(hdul[0].header['?JD-OBS']).split('=')[1].split('/')[0].strip())      # [d] MJD exposure start time 
    exptime = hdul[1].header['EXPOSURE']    # [s] eff. exposure duration
    snr = hdul[1].header['DER_SNR']         # Derived signal-to-noise ratio  
    
    ## Record redshift & temperature
    z = table['redshift'].value[0]
    Teff = table['Fe_H_Teff'].value[0]

    if Teff == 0: 
        Teff = 5000.01 # guess value of 5000.01 K if no Teff available

    ## Read in wavelength, flux, and fluxerr data

    wavelengths = redshift_correction(hdul[1].data['WAVE'][0], z) # Apply redshift correction
    rawflux = hdul[1].data['FLUX'][0]                             # raw because blackbody continuum will soon be subtracted
    fluxerr = hdul[1].data['SIGMA'][0]

    
    ## Subtract blackbody continuum emission from spectrum
    
    
    ## Save BB diagnostic plot
    
    
    ## Save reduced spectrum plot


    
    


if __name__ == "__main__":
    
    
    ## Read in table of IUE data
    table = Table.read(f'{datapath}/dataset.ecsv')

    
    ## Process each IUE spectrum
    for i, swpid in enumerate(table['obs_id']):

        ## Grab and reformat object name
        objname = table[i]['main_id'].replace(' ','')
        subfolder = f'{swpid}_{objname}' # e.g. swp55292_HD2454


        ## If analysis subfolder does not exist, create .../analysis/{swpid}_{objname}
        if not os.path.exists(f'{analysispath}/{subfolder}'):
            os.mkdir(f'{analysispath}/{subfolder}')


        ## Download data to ../data/subfolder
        if len(glob.glob(f'{datapath}/{swpid}.fits')) == 0:
            URL = table[i]['dataURL']
            response = request.urlretrieve(URL, f'{datapath}/{swpid}.fits')
    

    
    
    return 0