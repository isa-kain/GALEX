import numpy as np
import pandas as pd
import os, sys
from astropy.table import Table, vstack, hstack, join
from bokeh.plotting import figure, show, output_file
from bokeh.models import Span, ColumnDataSource, Slider, CustomJS, Button, Range1d
from bokeh.io import output_notebook, curdoc
from bokeh.layouts import column, row
from bokeh.events import ButtonClick

from bokeh.settings import settings
settings.py_log_level = "WARNING"
settings.log_level = "warn"

## Where should analysis results and data be saved to?
analysispath = '/Users/isabelkain/Desktop/GALEX/analysis'
datapath = '/Volumes/Seagate/seagate_backup/GALEX/data'


def get_pid():
    '''Get PID of bokeh server on port 5006'''

    output_stream = os.popen('lsof -i tcp:5006')
    outlines = output_stream.read().split('\n')
    
    for ol in outlines:
        if 'TCP *:wsm-server (LISTEN)' in ol:
            pid = ol.split(' ')[1]

    return pid


def closeport():
    pid = get_pid()
    os.system(f'kill -9 {pid}')


def BTcallback():
    
    ###########################
    ## Recover slider values ##
    ###########################
    
    ## Open linelist table
    linestable = pd.read_csv(f'{analysispath}/{swpid}_{objname}/linestable.csv') #Table.read(f'{datapath}/{swpid}_foundpeaks.ecsv')
    
    ## Read slider values
    slider_values = []
    user_slider_values = []
    
    for i in range(len(sliders)):
        slider_values.append( sliders[i].value )
    for i in range(len(user_sliders)):
        user_slider_values.append( user_sliders[i].value )
        
    ## If slider value = wavelengths.max(), ignore it
    for k in np.where(slider_values==wavelengths.max())[0]:
        slider_values[k] = 0.
    for k in np.where(user_slider_values==wavelengths.max())[0]:
        user_slider_values[k] = 0.
        
    
    ## Save slider values to lines table
    previous_peak_labels = linestable['Peak label'].values
    linestable['Measured peak'] = slider_values
    linestable['Peak label'] = [str(round(slider_values[j])) for j in range(len(slider_values))]
    
    ## Add user sliders as rows FIXME
    for j in range(len(user_slider_values)):
        r = pd.DataFrame({'Approx peak':[0],
                        'Measured peak':[user_slider_values[j]],
                        'Stddev':[0],
                        'Peak label':[str(round(user_slider_values[j]))],
                        'Spectrum':['?'],
                        'Confident?':[False]})
        linestable = pd.concat([linestable, r], axis=0, ignore_index=True)
        
        
    ###########################
    ## Save results          ##
    ###########################
    
    ## Change all previous NIST results with outdated peak labels FIXME
    for i, oldlabel in enumerate(previous_peak_labels):
        
        newlabel = linestable.loc[i, 'Peak label'] #str(round(slider_values[i]))

        ## If line has been thrown out, delete the line files
        if linestable.loc[i, 'Measured peak'] == 0:
            
            try: os.remove(f'{analysispath}/{swpid}_{objname}/lineID_tables/{oldlabel}.csv')
            except: pass
            
            try: os.remove(f'{analysispath}/{swpid}_{objname}/lineID_plots/{oldlabel}.png')
            except: pass

        ## If line has moved to a new position, rename it
        if oldlabel != newlabel:
            
            try:
                os.renames(f'{analysispath}/{swpid}_{objname}/lineID_tables/{oldlabel}.csv', 
                          f'{analysispath}/{swpid}_{objname}/lineID_tables/{newlabel}.csv')
            except: pass
            
            try:
                os.renames(f'{analysispath}/{swpid}_{objname}/lineID_plots/{oldlabel}.png', 
                          f'{analysispath}/{swpid}_{objname}/lineID_plots/{newlabel}.png')
            except: pass
    
    
    ## Unused lines are assigned linestable['Measured peak'] = 0. Throw out these rows. works now :)
    linestable = linestable[linestable['Measured peak'] != 0.]
    linestable.reset_index(drop=True, inplace=True)
    
    ## Save linestable
    linestable.to_csv(f'{analysispath}/{swpid}_{objname}/linestable.csv', index=False)
    
    
    
##################################
## Create interactive plot      ##
##################################
    

swpid = 'swp51800'
objname = 'HD216446'


## Read in data

spectrum = pd.read_csv(f'{datapath}/{swpid}.csv')
wavelengths = spectrum['Wavelengths'].values
flux = spectrum['Flux'].values
fluxerr = spectrum['Fluxerr'].values


## Read in lines
linestable = pd.read_csv(f'{analysispath}/{swpid}_{objname}/linestable.csv') # {swpid}_foundpeaks.ecsv


## Plot

p = figure(title=f"Spectrum for {objname} ({swpid})", 
           y_axis_label=r"Flux (erg/cm^2/s/A)", 
           x_axis_label=r"Wavelength (Angstrom)", sizing_mode='stretch_width', height=450)

p.line(wavelengths, flux, line_width=2)
p.varea(x=wavelengths, y1=flux-fluxerr, y2=flux+fluxerr, fill_color="gray", alpha=0.5)


## Add slider for each identified peak

sliders = []

for i in range(len(linestable)):
    
    if linestable.loc[i, 'Measured peak']==0:
        val = linestable.loc[i, 'Approx peak']
    else:
        val = linestable.loc[i, 'Measured peak']

    slider = Slider(start=wavelengths[0], end=wavelengths[-1], 
                    value=val, 
                    step=0.01, title=str(linestable.loc[i, 'Peak label']), width=600, align='center')
    span = Span(location=slider.value, dimension='height', line_color="#B488D3", line_dash='dashed')
    p.add_layout(span)

    callback = CustomJS(args=dict(span=span), code="""
        span.location = cb_obj.value
    """)
    slider.js_on_change('value', callback)

    sliders.append(slider)


    
## Add 10 extra sliders for user input:

user_sliders = []

for i in range(15):

    slider = Slider(start=wavelengths[0], end=wavelengths[-1], 
                    value=wavelengths[-1], 
                    step=0.01, title=f'User line {i+1}', width=600, align='center')
    span = Span(location=slider.value, dimension='height', line_color="#2d8659", line_dash='dashdot')
    p.add_layout(span)

    callback = CustomJS(args=dict(span=span), code="""
        span.location = cb_obj.value
    """)
    slider.js_on_change('value', callback)

    user_sliders.append(slider)
    

## Add button to save results
button = Button(label='Save inputs', button_type='success', align='center')
button.on_event(ButtonClick, BTcallback)

closebutton = Button(label='Close', button_type='success', align='center')
closebutton.on_event(ButtonClick, closeport)

## Format and show

row2 = row(column(*sliders, sizing_mode='scale_width'), 
           column(*user_sliders, sizing_mode='scale_width'), sizing_mode='scale_width')  
rowB = row(button, closebutton, align='center')
layout = column(p, rowB, row2, sizing_mode='scale_width')

curdoc().add_root(layout)
show(layout)
