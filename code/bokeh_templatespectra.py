import numpy as np
import pandas as pd
import os, sys
from astropy.table import Table, vstack, hstack, join
from bokeh.plotting import figure, show, output_file
from bokeh.models import Span, ColumnDataSource, Slider, CustomJS, Button, Range1d, CheckboxGroup, Ray
from bokeh.io import output_notebook, curdoc
from bokeh.layouts import column, row
from bokeh.events import ButtonClick
import glob
from matplotlib.colors import rgb2hex
import matplotlib as mpl
import matplotlib.pyplot as plt

from bokeh.settings import settings
settings.py_log_level = "WARNING"
settings.log_level = "warn"

## Where should analysis results and data be saved to?
analysispath = '/Users/isabelkain/Desktop/GALEX/analysis'
datapath = '/Volumes/Seagate/seagate_backup/GALEX/data'


def closeport():
    '''Get PID of bokeh server on port 5006'''

    output_stream = os.popen('lsof -i tcp:5006')
    outlines = output_stream.read().split('\n')
    
    for ol in outlines:
        if 'TCP *:wsm-server (LISTEN)' in ol:
            pid = ol.split(' ')[1]
            
    os.system(f'kill -9 {pid}')
    
    
def update(attr, old, new):
    '''Change visibility of template spectrum based on checkbox status'''
    
    for i, r in enumerate(rays):
        r.visible = i in checkbox_group.active
        
def clearall():
    '''Hide all template spectra'''
    
    for i, r in enumerate(rays):
        r.visible = False
    
    checkbox_group.active = []
    
def save_selection():
    
    active = checkbox_group.active
    active_elements = [elements[i] for i in active]
    
    selected_elements = pd.DataFrame(data=active_elements, columns=['Elements'])
    selected_elements.to_csv(f'{analysispath}/{swpid}_{objname}/lineID_tables/selected_elements.csv', index=False)
  
    

##################################
## Create interactive plot      ##
##################################
    

swpid = 'swp54894'
objname = '*gamMic'

## Read in data
spectrum = pd.read_csv(f'{datapath}/{swpid}.csv')
wavelengths = spectrum['Wavelengths'].values
flux = spectrum['Flux'].values
fluxerr = spectrum['Fluxerr'].values

## How many element templates?
tmplt = glob.glob(f'{analysispath}/{swpid}_{objname}/lineID_tables/*template.csv')
elements = [tmplt[i].split('/')[-1].split('_')[0] for i in range(len(tmplt))]


########################
## Plot IUE spectrum  ##
########################

p = figure(title=f"Spectrum for {objname} ({swpid})", 
           y_axis_label=r"Flux (erg/cm^2/s/A)", 
           x_axis_label=r"Wavelength (Angstrom)", sizing_mode='stretch_width', height=450)

p.line(wavelengths, flux, line_width=2)
p.varea(x=wavelengths, y1=flux-fluxerr, y2=flux+fluxerr, fill_color="gray", alpha=0.5)


## Add shaded spans around range of NIST queries FIXME
linestable = pd.read_csv(f'{analysispath}/{swpid}_{objname}/linestable.csv')

for peak in linestable['Measured peak']:
    p.harea([peak-5., peak-5.], [peak+5., peak+5.], [flux.min()*1.8, flux.max()*1.5], fill_alpha=0.3, fill_color='gray')

    
############################
## Plot template spectra  ##
############################

## Make color cycler
cmap = plt.get_cmap('turbo')(np.linspace(0.0, 1.0, len(elements)))
colors = [rgb2hex(cmap[i]) for i in range(len(elements))]

## For each template (FOR NOW JUST ONE), add checkbox and plot lines

rays = []

for i, el in enumerate(elements): # e.g. 'Na II'

    template = pd.read_csv(f'{analysispath}/{swpid}_{objname}/lineID_tables/{el}_template.csv')

    linestyle = np.array(['solid']*len(template))               # solid for pipeline matched lines
    linestyle[~template['Pipeline matched'].values] = 'dotted'  # dotted for lines from NIST not found in spectrum
    
    alpha = [(float(template.loc[i, 'Rel.'])/template['Rel.'].max())**.25 for i in range(len(template))]

    r = p.ray(x=template['Observed'].values, y=[flux.min()]*len(template), length=0, angle=90, 
              angle_units='deg', line_dash=linestyle, line_alpha=alpha, color=colors[i])
    
    rays.append(r)


checkbox_group = CheckboxGroup(labels=elements, active=list(np.arange(len(elements))))
checkbox_group.on_change('active', update)

clearbutton = Button(label='Hide all templates', button_type='success')
clearbutton.on_event(ButtonClick, clearall)

savebutton = Button(label='Save selection', button_type='success')
savebutton.on_event(ButtonClick, save_selection)

quitbutton = Button(label='Close', button_type='success')
quitbutton.on_event(ButtonClick, closeport)

layout = row(p, column(savebutton, clearbutton, quitbutton, checkbox_group), sizing_mode='scale_width') # checkbox_group
curdoc().add_root(layout)