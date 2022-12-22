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


def clean_spectrum(table):
    '''
    Read in, apply redshift correction, and subtract blackbody continuum from spectrum.
    Input:
    table [astropy Table]: single row of full XMatched dataset corresponding to the spectrum being prepared
    
    Returns:
    ???
    '''
    ######################
    ## Read in raw data ##
    ######################
    
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
    
    hdul.close()

    
    ###########################
    ## Blackbody subtraction ##
    ###########################
    
    ## Mask most features to let curve_fit see the underlying emission

    med = np.median(rawflux)
    std = np.std(rawflux)
    mask = np.logical_and((rawflux <= med + .5*std), (rawflux >= med - .5*std)) 
    
    x = wavelengths[mask]
    y = rawflux[mask]
    yerr = fluxerr[mask]

    ## Fit blackbody curve. If Teff known, then fix for curve_fit, else let vary
    
    if Teff == 5000.01:
        popt, pcov = curve_fit(fitBB_fitTeff, x, y, p0=[Teff, 0., 0.],
                               sigma=yerr, absolute_sigma=True,
                               bounds=((4000., 0, 0), (9000., y.max(), np.inf))) 
        
        Teff = popt[0]

    else:
        popt, pcov = curve_fit(lambda x, a, b: fitBB_fixedTeff(x, Teff, a, b), x, y, 
                               sigma=yerr, absolute_sigma=True,
                               bounds=((-np.inf, 0), (np.inf, np.inf))) 

    ## Plot masked spectrum and scaled blackbody curve, save as diagnostic

    plt.plot(x, y, label='Spectrum')

    if Teff == 5000.01:
        plt.plot(wavelengths, fitBB_fitTeff(wavelengths, popt[0], popt[1], popt[2]), 
                 label=f'Blackbody fit:\n{popt[1]:0.2e}func + {popt[2]:0.2e}\nTeff = {popt[0]}')
    else:
        plt.plot(wavelengths, fitBB_fixedTeff(wavelengths, Teff, popt[0], popt[1]), 
                 label=f'Blackbody fit:\n{popt[0]:0.2e}func + {popt[1]:0.2e}\nTeff = {Teff}')

    plt.fill_between( x, y-yerr, y+yerr, alpha=0.5 )
    plt.legend()

    plt.savefig(f'{analysispath}/{swpid}_{objname}/{swpid}_blackbody.png', bbox_inches='tight')
    plt.close()
    
    
    ## Subtract BB curve

    flux = rawflux - fitBB_fixedTeff(wavelengths, Teff, popt[0], popt[1])

    
    ## Plot BB-subtracted flux

    plt.plot( wavelengths, flux, color='k', label='Blackbody-subtracted' )
    plt.fill_between(wavelengths, flux + fluxerr, flux - fluxerr, alpha=0.5, color='gray')

    plt.xlabel(r'Wavelength ($\AA$)', fontsize=14)
    plt.ylabel(r'Flux (erg/cm$^2$/s/$\AA$)', fontsize=14)

    plt.suptitle(f'{starname}', fontsize=14)
    plt.title(f'{dispersion}-DISP {camera}, {spclass}, SNR {snr}')

    plt.savefig(f'{analysispath}/{swpid}_{objname}/{swpid}.png', bbox_inches='tight')
    plt.close()
    
    
    ## Save cleaned spectrum

    specDF = pd.DataFrame(data=np.array([wavelengths, flux, fluxerr]).T, columns=['Wavelengths', 'Flux', 'Fluxerr'])
    specDF.to_csv(f'{datapath}/{swpid}.csv', index=False)

    
    return wavelengths, flux, fluxerr, spclass, snr, Teff


def find_lines(wavelengths, flux, fluxerr, diagnostic_plots=True):
    '''
    Identify peaks in stellar spectrum.
    Inputs:
    wavelengths [arr]: 
    flux [arr]: 
    fluxerr [arr]: 
    diagnostic_plots [bool]: if true, saves plot of fit results for each individual line
    
    Returns:
    '''
    
    ## Create a Spectrum1D object, subtracting smoothed spectrum
    
    w = 5
    raw_spectrum = Spectrum1D(flux = flux * (u.erg/u.cm**2/u.s/u.AA), spectral_axis = wavelengths * u.AA)
    spectrum = trapezoid_smooth(raw_spectrum, w)

    
    ## Define the uncertainty of the spectrum using everything redward of the H line FIXME
    
    noise_region = SpectralRegion(np.floor(wavelengths[0])*u.AA, np.ceil(wavelengths[-1])*u.AA)
    spectrum = noise_region_uncertainty(spectrum, noise_region)


    ## Find lines using a noise threshold

    with warnings.catch_warnings():

        warnings.simplefilter('ignore')
        thresh = np.median(np.abs(spectrum.flux.value)) + MAD(spectrum.flux.value)
        lines = find_lines_derivative(spectrum, thresh*(u.erg/u.cm**2/u.s/u.AA)) 


    ## Plot identified lines

    plt.figure(figsize=(12,4))

    plt.plot( wavelengths, flux, c='k')
    plt.fill_between(wavelengths, flux + fluxerr, flux - fluxerr, alpha=0.5, color='gray')
    plt.hlines( thresh, wavelengths.min(), wavelengths.max(), ls='--' )

    plt.ylim(0.2*flux.min(), 30*np.median(np.abs(flux)))
    ymin, ymax = plt.ylim()

    for l in lines:

        if l['line_type']=='emission':
            plt.vlines(l['line_center'].value, ymin, ymax, alpha=0.6, color='cornflowerblue')
        elif l['line_type']=='absorption':
            plt.vlines(l['line_center'].value, ymin, ymax, alpha=0.6, color='goldenrod')
            
            
    ## Save plot
    
    if not os.path.exists(f'{analysispath}/{swpid}_{objname}/linefit_plots'):
        os.mkdir(f'{analysispath}/{swpid}_{objname}/linefit_plots')

    plt.savefig(f'{analysispath}/{swpid}_{objname}/linefit_plots/found_peaks.png')
    
    
    return lines





def fit_lines(wavelengths, flux, fluxerr, lines, diagnostic_plots=True):
    '''
    Fit Gaussian profile to peaks in stellar spectrum to determine which are true lines.
    Inputs:
    wavelengths [arr]: 
    flux [arr]: 
    fluxerr [arr]: 
    lines [astropy QTable]:
    diagnostic_plots [bool]: if true, saves plot of fit results for each individual line
    
    Returns:
    linestable [pandas DF]: table of successfully fitted lines
    '''
    
    ## Create new columns in lines

    lines['peak_fit'] = np.zeros(len(lines)) * u.AA
    lines['peak_std'] = np.zeros(len(lines)) * u.AA


    ## Make copy of spectrum to mask lines as we fit them

    search_spectrum = Spectrum1D(flux=raw_spectrum.flux, spectral_axis=raw_spectrum.spectral_axis) # use unsmoothed, smoothing shifts peak locations enough to be troublesome
    search_spectrum.mask = [False]*len(search_spectrum.data)
    
    
    ## Fit Gaussian to each line

    for i, cen in enumerate(lines[lines['line_type']=='emission']['line_center'].value):
        
        #########################
        ## Gaussian fit        ##
        #########################

        ## Set window around line center
        margin = 12. # AA
        ulim = cen - margin
        hlim = cen + margin

        # have to do this stupid shit because Spectrum1D doesn't ACTUALLY mask its data
        xmask = search_spectrum.spectral_axis.value[~search_spectrum.mask] 
        ymask = search_spectrum.flux.value[~search_spectrum.mask]
        yerrmask = fluxerr[~search_spectrum.mask]

        # Trim x and y arrays down to window around line
        trim = (xmask >= ulim) & (xmask < hlim)
        x = xmask[trim]
        y = ymask[trim]
        yerr = yerrmask[trim]

        window = Spectrum1D(flux=y*(u.erg/u.cm**2/u.s/u.AA), spectral_axis=x*u.AA)

        ## Initialize Gaussian model and fitter
        fitter = LevMarLSQFitter()
        g_init = models.Gaussian1D(amplitude=y.max() * (u.erg/u.cm**2/u.s/u.AA), 
                                   mean=cen * u.AA, 
                                   stddev=3.*u.AA)

        ## Fit model to window around line 
        g_fit = fit_lines(window, g_init, fitter=fitter)
        y_fit = g_fit(x * u.AA)


        #########################
        ## Acceptance criteria ##
        #########################

        ## Check if fit succeeded

        crit = [fitter.fit_info['ierr'] not in [1, 2, 3, 4], 
                np.abs( cen - g_fit.mean.value ) > 3., 
                g_fit.stddev.value <= 0.01,
                g_fit.stddev.value >= 4.0] ## is this good enough? is there a more elegant metric?
        fail = np.any(crit)

        
        if not fail:

            ## Mask line in search_spectrum
            line_center = g_fit.mean.value # AA
            line_stddev = g_fit.stddev.value

            line_lowlim = line_center - 3*line_stddev
            line_uplim = line_center + 3*line_stddev

            hideline = (search_spectrum.spectral_axis.value >= line_lowlim) & (search_spectrum.spectral_axis.value < line_uplim)
            oldmask = search_spectrum.mask

            search_spectrum.mask = np.logical_or(hideline, oldmask)
            print('Mask check: ', np.sum(hideline), np.sum(np.logical_or(hideline, oldmask)), np.sum(search_spectrum.mask) )


            ## Save peak wavelength
            j = np.argmin(np.abs(lines['line_center'] - cen * u.AA))
            lines[j]['peak_fit'] = g_fit.mean.value * u.AA
            lines[j]['peak_std'] = g_fit.stddev.value * u.AA


        #########################
        ## Plot results        ##
        #########################

        if diagnostic_plots:

            if fail: status = 'fail'
            else: status = 'pass'

            fig = plt.figure()
            gs = fig.add_gridspec(2, 1,  height_ratios=(3, 1))

            # Plot isolated line and fit
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])

            ax1.plot( x, y, c='k', label='Spectrum' )
            ax1.fill_between( x, y-yerr, y+yerr, alpha=0.4, color='gray', label='Uncertainty' )
            ax1.plot( x, y_fit, label='Line fit' )
            ymin, ymax = ax1.set_ylim()

            ax1.vlines(cen, ymin, ymax, ls=':', color='g', label='Line center' )
            ax1.vlines(g_fit.mean.value, ymin, ymax, ls='--', color='g', label='Peak fit' )

            ax1.set_title(f'index {i} @ {cen:0.2f} AA: {status}')
            ax1.legend()
            ax1.set_xlabel(r'Wavelength ($\AA$)', fontsize=14)
            ax1.set_ylabel(r'Flux (erg/cm$^2$/s/$\AA$)', fontsize=14)

            # Plot full spectrum with line annotated
            ax2.plot( wavelengths, flux, c='k' )
            ax2.fill_between( wavelengths, flux + fluxerr, flux - fluxerr, alpha=0.5, color='gray' )
            ax2.hlines( thresh, wavelengths.min(), wavelengths.max(), ls='--' )

            ax2.set_ylim(0.2*flux.min(), 30*np.median(np.abs(flux)))
            ymin, ymax = ax2.set_ylim()

            ax2.vlines(g_fit.mean.value, ymin, ymax, alpha=0.6, color='cornflowerblue', label='Flux threshold')
            ax2.legend()
            ax2.set_xlabel(r'Wavelength ($\AA$)', fontsize=14)


            ## Save figure
            if not os.path.exists(f'{analysispath}/{swpid}_{objname}/linefit_plots'):
                os.mkdir(f'{analysispath}/{swpid}_{objname}/linefit_plots')

            plt.savefig(f'{analysispath}/{swpid}_{objname}/linefit_plots/{i}_{round(cen)}_{status}.png')
            plt.close()


    ## Record successfully fit lines
    
    goodlines = lines[lines['peak_fit']!=0.0*u.AA]
    
    
    ## Make plot of full spectrum with line positions annotated
    
    plt.figure(figsize=(12,4))
    plt.plot(wavelengths, flux, color='k', zorder=5)
    plt.fill_between(wavelengths, flux-fluxerr, flux+fluxerr, alpha=0.5, color='gray', zorder=5)
    plt.hlines( thresh, wavelengths.min(), wavelengths.max(), ls='--' )

    plt.ylim(0.5*flux.min(), 50*np.median(np.abs(flux)))
    ymin, ymax = plt.ylim()

    for line in goodlines:
        plt.vlines(line['peak_fit'].value, ymin, ymax)


    ## Save full spectrum figure
    
    if not os.path.exists(f'{analysispath}/{swpid}_{objname}/linefit_plots'):
        os.mkdir(f'{analysispath}/{swpid}_{objname}/linefit_plots')

    plt.savefig(f'{analysispath}/{swpid}_{objname}/linefit_plots/all_lines.png')
    plt.close()

    ## Save table of found lines
    
    columns=['NIST peak', 'Measured peak', 'Stddev', 'Peak label', 'Spectrum', 'Confident?']
    rows = np.array([goodlines['line_center'], goodlines['peak_fit'], goodlines['peak_std'], 
                     np.round(goodlines['line_center'].value).astype(int),np.full(len(goodlines), '?'), 
                     np.zeros(len(goodlines))]).T

    linestable = pd.DataFrame(columns=columns, data=rows)
    linestable.to_csv(f'{analysispath}/{swpid}_{objname}/linestable.csv', index=False)

    
    return linestable



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
            
            
        ## Clean up spectrum (apply redshift correction, subtract blackbody continuum)
        row = table[i]
        wavelengths, flux, fluxerr, spclass, snr, Teff = clean_spectrum(row)

    
    
    return 0