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
from multiprocessing import Pool
import istarmap

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
from astropy.modeling.fitting import LevMarLSQFitter
from specutils.fitting import find_lines_threshold, find_lines_derivative, fit_lines
from specutils.manipulation import noise_region_uncertainty, trapezoid_smooth
from specutils import Spectrum1D, SpectralRegion
from astropy.nddata import NDUncertainty
from scipy.optimize import curve_fit

from astroquery.simbad import Simbad
from astroquery.nist import Nist
from astroquery.xmatch import XMatch
from astroquery.vizier import Vizier
from astroquery.mast import Observations

from pipeline import queryNIST ## my function!!!


## Where should analysis results and data be saved to?
analysispath = '/Users/isabelkain/Desktop/GALEX/analysis'
datapath = '/Volumes/Seagate/seagate_backup/GALEX/data'
codepath = '/Users/isabelkain/Desktop/GALEX/code'

scriptname = 'bokehplot.py'
scriptpath = f'{codepath}/{scriptname}'

## Helper functions
angstrom2meter = 10**-10
joules2ergs = 10**7



def user_peaks(wavelengths, flux, fluxerr):
    '''
    Since lines are hard to fit because of intrinsic shape and low SNR, 
    ask user to identify lines by eye in reduced spectrum.
    Inputs:
    wavelengths [arr]: 
    flux [arr]: 
    fluxerr [arr]: 
    
    Returns:
    linestable [pd DataFrame]: table of peaks found by specutils + user

    '''
    
    ## Change swpid, objname, starname in bokeh script
    replace_swpid(scriptpath, swpid, objname)
    
    ## Call script to host bokeh server, where user adjusts line-finding results
    os.system(f'bokeh serve --show {scriptname}')

    ###########################
    ## Plot results          ##
    ###########################
        
    linestable = pd.read_csv(f'{analysispath}/{swpid}_{objname}/linestable.csv')
    
    ## Make plot of full spectrum with line positions annotated
    
    plt.figure(figsize=(12,4))
    plt.plot(wavelengths, flux, color='k', zorder=5)
    plt.fill_between(wavelengths, flux-fluxerr, flux+fluxerr, alpha=0.5, color='gray', zorder=5)

    plt.ylim(0.5*flux.min(), 50*np.median(np.abs(flux)))
    ymin, ymax = plt.ylim()

    for i in range(len(linestable)):
        plt.vlines(linestable.loc[i, 'Measured peak'], ymin, ymax, color='#B488D3', lw=3, ls='--')

    ## Save full spectrum figure
    
    if not os.path.exists(f'{analysispath}/{swpid}_{objname}/linefit_plots'):
        os.mkdir(f'{analysispath}/{swpid}_{objname}/linefit_plots')

    plt.savefig(f'{analysispath}/{swpid}_{objname}/linefit_plots/all_lines.png')
    plt.close('all')
    
    return linestable


def replace_swpid(scriptpath, swpid, objname):
    '''
    Write swpid, objname, and starname into script that runs bokeh line finder plot.
    '''
    
    #Create temp file
    fh, abs_path = mkstemp()
    
    with os.fdopen(fh,'w') as new_file:
        
        with open(scriptpath) as old_file:
            
            content_list = old_file.readlines()
            
            for i, line in enumerate(content_list):
                if 'swpid = ' in line:
                    content_list[i] = f'swpid = \'{swpid}\'\n'
                if 'objname = ' in line:
                    content_list[i] = f'objname = \'{objname}\'\n'

                new_file.write(content_list[i])
            
                
    #Copy the file permissions from the old file to the new file
    shutil.copymode(scriptpath, abs_path)
    
    #Remove original file
    os.remove(scriptpath)
    
    #Move new file
    shutil.move(abs_path, scriptpath)
    
    return 0



def runNISTqueries(row, analysispath, datapath):
    '''
    Calls queryNIST on a given row in database table, used for multiprocessing of this step.
    '''

    ## Grab and reformat object name
    swpid = row['obs_id']
    starname = row['main_id']
    objname = starname.replace(' ','')
    subfolder = f'{swpid}_{objname}' # e.g. swp55292_HD2454

    ## Read in saved spectrum
    spectrum = pd.read_csv(f'{datapath}/{swpid}.csv')
    wavelengths = spectrum['Wavelengths'].values
    flux = spectrum['Flux'].values
    fluxerr = spectrum['Fluxerr'].values

    ## Read in linestable
    linestable = pd.read_csv(f'{analysispath}/{swpid}_{objname}/linestable.csv')

    ## Query NIST for atomic lines near each identified spectral line
    if row['Peaks verified']:
        queryNIST(linestable, wavelengths, flux, fluxerr, swpid, objname)

    return 0



if __name__ == "__main__":
    
    
    ## Read in table of IUE data
    table = Table.read(f'{analysispath}/dataset.ecsv')

    ## User approves spectral lines for each spectrum
    for i, swpid in enumerate(tqdm(table['obs_id'])):
        
        ## Grab and reformat object name
        starname = table[i]['main_id']
        objname = starname.replace(' ','')
        subfolder = f'{swpid}_{objname}' # e.g. swp55292_HD2454
        
        ## Read in saved spectrum
        spectrum = pd.read_csv(f'{datapath}/{swpid}.csv')
        wavelengths = spectrum['Wavelengths'].values
        flux = spectrum['Flux'].values
        fluxerr = spectrum['Fluxerr'].values

        if (table['Peaks verified'][i]==False) or (table['Rerun verification'][i]==True):

            ## Have user identify spectral lines
            linestable = user_peaks(wavelengths, flux, fluxerr)
            table['Peaks verified'][i] = True
            table.write(f'{analysispath}/dataset.ecsv', overwrite=True)


    print(f'Peaks in {len(table)} spectra verified. Querying NIST for matching lines.')


    ## Run results through queryNIST using multiprocessing
    args = [(table[i], analysispath, datapath) for i in range(len(table))]

    with Pool() as pool:        
        for _ in tqdm(pool.istarmap(runNISTqueries, args), total=len(args)):
            pass
