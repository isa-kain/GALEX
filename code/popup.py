import numpy as np
import pandas as pd
import PySimpleGUI as sg
from bokeh.plotting import figure, show
from bokeh.models import Span
from astropy.table import Table, vstack, hstack, join

analysispath = '/Users/isabelkain/Desktop/GALEX/analysis'
datapath = '/Users/isabelkain/Desktop/GALEX/data'

def popup(swpid, objname, cen):
    '''
    Generate popup window for user to ID spectral line.
    Inputs:
    swpid [str]: SWP ID of observation, e.g. swp54894
    objname [str]: colloquial name of object no spaces, e.g. *gamMic
    cen [int]: rounded wavelength of line being ID'd, e.g. 1216
    '''
    
    sg.theme('BlueMono')
    modal=True
    
    ## Define paths to line snapshot and NIST line results
    imgpath = f'{analysispath}/{swpid}_{objname}/lineID_plots/{linewav}.png'
    tablepath = f'{analysispath}/{swpid}_{objname}/lineID_tables/{linewav}.csv'

    ## Load table of good lines identified by fitting routine
    linetable = pd.read_csv(f'{analysispath}/{swpid}_{objname}/linestable.csv')

    ## Find row matching line indicated by cen
    loc = linetable.index[linetable['Peak label'] == cen].tolist()[0]
    linewav = round(linetable.loc[loc,'Measured peak'])
    print(linewav)

    ## Load table of possible lines queried from NIST
    tbl = pd.read_csv(tablepath)
    tbl_list = tbl.values.tolist()
    tbl_cols = list(tbl.columns.values)


    ## Define layout of window
    col1 = [[sg.Image(imgpath)],
            [sg.Text('Spectrum:')],
            [sg.Input(default_text = f"{linetable.loc[loc,'Spectrum']}", key='-SPECTRUM-')],
            [sg.Checkbox('Confident?', default=True, key='-CONFIDENT-')],
            [sg.Button('Ok'), sg.Button('Cancel')] ]
    col2 = [sg.Table(tbl_list, headings = tbl_cols)]
    layout = [[col1, col2]]

    # Create the Window
    window = sg.Window('Spectral line identification', layout)

    while True:
        event, values = window.read()
        print(event, values)
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