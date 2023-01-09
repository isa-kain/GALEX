import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import re, os
import glob
import warnings
from tqdm import tqdm
import PySimpleGUI as sg

from astropy.io import fits
import astropy.units as u
import astropy.constants as const
import astropy.coordinates as coord
from astropy.table import Table, vstack, hstack, join

matplotlib.use("TKAgg")


## Where should analysis results and data be saved to?
analysispath = '/Users/isabelkain/Desktop/GALEX/analysis'
datapath = '/Volumes/Seagate/seagate_backup/GALEX/data'
codepath = '/Users/isabelkain/Desktop/GALEX/code'

## Helper functions
angstrom2meter = 10**-10
joules2ergs = 10**7



def line_popup(swpid, objname, starname, cen, linetable):
    '''
    Generate popup window for user to ID spectral line.
    Inputs:
    swpid [str]: SWP ID of observation, e.g. swp54894
    objname [str]: colloquial name of object no spaces, e.g. *gamMic
    cen [int]: rounded wavelength of line being ID'd, e.g. 1216
    linetable [QTable]: table of found peaks with column for spectrum identification
    '''
    
    sg.theme('BlueMono')
    modal=True
    
    ## Define paths to line snapshot and NIST line results
    imgpath = f'{analysispath}/{swpid}_{objname}/lineID_plots/{str(cen)}.png'
    tablepath = f'{analysispath}/{swpid}_{objname}/lineID_tables/{str(cen)}.csv'

    ## Load table of good lines identified by fitting routine
    linetable = pd.read_csv(f'{analysispath}/{swpid}_{objname}/linestable.csv')

    ## Find row matching line indicated by cen
    loc = linetable.index[linetable['Peak label'] == cen].tolist()[0]

    ## Load table of possible lines queried from NIST
    tbl = pd.read_csv(tablepath)
    tbl_list = tbl.values.tolist()
    tbl_cols = list(tbl.columns.values)


    ## Define layout of window
    col1 = [[sg.Text('Spectrum:')],
            [sg.Input(default_text = f"{linetable.loc[loc,'Spectrum']}", key='-SPECTRUM-')],
            [sg.Checkbox('Confident?', default=True, key='-CONFIDENT-')],
            [sg.Button('Ok'), sg.Button('Cancel')] ]
    col2 = [[sg.Table(tbl_list, headings = tbl_cols, font='Helvetica 14')]]
    layout = [[sg.Image(imgpath)], [col1, col2]]

    # Create the Window
    window = sg.Window(f'Spectral line identification {swpid}', layout)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            break

        if event == 'Ok':
            ## Record spectrum and confidence data
            linetable.loc[loc, 'Spectrum']   = values['-SPECTRUM-']
            linetable.loc[loc, 'Confident?'] = values['-CONFIDENT-']
            break

    window.close()

    ## Write changes
    linetable.to_csv(f'{analysispath}/{swpid}_{objname}/linestable.csv', index=False)
    
    return 0


def plot_spectrum(swpid, objname, linetable):
    '''
    Save png of full spectrum of object with lines annotated.
    Inputs:
    swpid [str]: SWP ID of observation, e.g. swp54894
    objname [str]: colloquial name of object no spaces, e.g. *gamMic
    linetable [QTable]: table of found peaks with column for spectrum identification
    '''
    
    ## Load spectrum
    spectrum = pd.read_csv(f'{datapath}/{swpid}.csv')
    
    ## Load table of good lines identified by fitting routine
    centers = linetable['Peak label'].values
    peakvals = linetable['Measured peak'].values
    linespecs = linetable['Spectrum'].values

    for i in range(len(linespecs)):
        if (linespecs[i]!='?') & (linetable['Confident?'][i]==False):
            linespecs[i] = linespecs[i] + '?'


    ## Plot spectrum
    fig = plt.figure(figsize=(12,4))
    plt.plot(spectrum['Wavelengths'], spectrum['Flux'])
    plt.fill_between(spectrum['Wavelengths'], spectrum['Flux']-spectrum['Fluxerr'], 
                     spectrum['Flux']+spectrum['Fluxerr'], alpha=0.5)

    ymin, ymax = plt.ylim()

    for i, peak in enumerate(peakvals):
        plt.axvline(peak, color='#B488D3', ls='--')
        plt.annotate( linespecs[i], xy=(peak+5, 0.85*ymax), color='#713e95', fontsize=14 )

    plt.savefig(f'{analysispath}/{swpid}_{objname}/analyzed_spectrum.png')
    plt.close('all')

    return 0
    
    
    
def spectrum_popup(swpid, objname, starname, linetable):
    '''
    Generate popup window that shows full spectrum of given object, with found lines annotated.
    Inputs:
    swpid [str]: SWP ID of observation, e.g. swp54894
    objname [str]: colloquial name of object no spaces, e.g. *gamMic
    starname [str]: colloquial name of object, e.g. * gam Mic
    linetable [QTable]: table of found peaks with column for spectrum identification
    '''

    modal=False

    
    ## Plot figure
    imgpath = f'{analysispath}/{swpid}_{objname}/analyzed_spectrum.png'
    plot_spectrum(swpid, objname, linetable)


    ## Make buttons
    centers = np.sort(linetable['Peak label'].values)
    labels = np.array([fr'{centers[i]} A' for i in range(len(centers))])
    buttons = []
    
    for l in labels:
        b = sg.Button(l)
        buttons.append(b)
        

    ## Make image
    img = sg.Image(imgpath, key='-SPECTRUM-')


    ## Window layout
    layout = [[img], [sg.Text('Identify lines:'), *buttons], [sg.Button('Close'), sg.Button('Cancel')]]

    
    ## Make window -- python is breaking here
    window = sg.Window(f'{starname} / {swpid} spectrum', layout, finalize=True, 
                       element_justification='center', font='Helvetica 18')
    
    while True:

        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'Close':
            break
        if event == 'Cancel':
            return -1

        if np.any(event == labels):

            ## Call popup for individual line
            cen = int(re.sub(" [a-zA-Z]", "", event))
            line_popup(swpid, objname, starname, cen, linetable)

            ## Replot spectrum
            plot_spectrum(swpid, objname, linetable)

            ## Update spectrum in window
            img.update(imgpath)

    return 0





if __name__ == "__main__":
    
    
    ## Read in table of IUE data
    table = Table.read(f'{datapath}/dataset.ecsv')
    

    ## Process each IUE spectrum
    for i, swpid in enumerate(tqdm(table['obs_id'])):
        
        ## Grab and reformat object name
        starname = table[i]['main_id']
        objname = starname.replace(' ','')
        
        ## Proceed if this object is marked for line identification
        if (table['Peaks identified'][i]==False) or (table['Rerun identification'][i]==True):

            ## Read in table of lines output by both fit_peaks and user_peaks
            try:
                linetable = pd.read_csv(f'{analysispath}/{swpid}_{objname}/linestable.csv')
            except:
                print(f'Linestable does not yet exist for {swpid} / {objname}. Please run pipeline.py to generate.')
                continue
                
            ## Check if queryNIST has been run yet (& line candidate plots generated for popups)
            if os.path.exists(f'{analysispath}/{swpid}_{objname}/lineID_plots'):

                ## If user identification is successfull, mark that peaks have been identified
                res = spectrum_popup(swpid, objname, starname, linetable)
                if res != -1:
                    table['Peaks identified'][i] = True
                    table.write(f'{datapath}/dataset.ecsv', overwrite=True)
                    
            else:
                print(f'NIST has not been queried for {swpid} / {objname}. Please run pipeline.py or verify_peaks.py')

    
    
    
    
