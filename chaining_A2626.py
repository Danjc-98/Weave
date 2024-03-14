#!/usr/bin/env python
# coding: utf-8

############ Overview: Chains the configuration of a set of pre-labelled, pre-ordered pointings, adapted from the Workflow.  Minimizes target priorities of configured targets after each configuration. Alternates between plate A and plate B. Current setup is designed to work on Captain, using my paths.

############ Requirements: Python 3. Packages -- numpy, pandas, os, sys, glob, subprocess, astropy
############             : WorkFlow -- Latest version from BitBucket WEAVE SWG, make sure to use the correct branch (e.g. SV, master etc). Clear instructions on how to clone the directory available on https://ingbitbucket.ing.iac.es/projects/WVSWG
############             : Configure -- Latest version from Confluence. NB: if running on Captain, need the CentOS version
############             : Directory -- needs to be ran in the same directory as obviewer.py, protofielding.py etc, e.g. .../mos/examples/        -- I haven't tested this for different directories yet 
############             : Files -- input fits file, pointing position files (one file per pointing, pre-labelled), 

############ inputs: low priority, high priority, updated low priority, updated low priority, number of fields, cluster name, field name prefix

import numpy as np
import pandas as pd
from numpy import genfromtxt
import os
import sys
import glob
import subprocess
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits
import warnings
from os.path import dirname, join

from bokeh.io import output_notebook, show, curdoc, save
from bokeh.layouts import gridplot, layout, column, row
from bokeh.palettes import Reds4, Spectral4, Blues9, viridis
from bokeh.plotting import figure, output_file
from bokeh.models import ColumnDataSource, Div, Select, RadioButtonGroup
from bokeh.transform import factor_cmap, factor_mark

warnings.simplefilter('ignore', category=RuntimeWarning)
pd.options.mode.chained_assignment = None  # default='warn'

sys.path.append('../')
from protofielding import MOSDriverCat, MOSTrackingCat, MOSOBXML
path_to_configure_tools = '/home/ppxdc1/configure_versions/configure_3.4.5/' ### set as your directory

os.environ["MKL_NUM_THREADS"] = "4" ### specifies how many threads captain can use
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

def config(input_fits,input_field, xml_2_config, WEAVE_clus, plate):
    """ configures targets
        
        Parameters:
        
            input_fits is the input fits file
            input_field is the unique field name
            xml_2_config is xml file to configure. note that this is the name of the xml file that is generated in the WF
            WEAVE_clus is the name of the cluster
            plate is PLATE_A or PLATE_B
            both input_fits/input_field files need suffix (.fits/.csv)
          
        Returns:
            Configured xml
        """
    hetdex = MOSDriverCat(f'input/fits_files/{input_fits}')

    field_list = Table.read(f'input/Tiles/{input_field}')
    field_list.show_in_notebook()

    for field in field_list:
        hetdex.process_ob(field['RA'], field['DEC'],
                      field['NAME'],
                      '11331', # Currently the only option in the HETDEX catalog
                      'IBCEB', # Fixed for all MOS WL targets
                      plate,
                      num_guide_stars_request=8,
                      output_dir=f'Output/{WEAVE_clus}')

        
    fields_to_configure = [f'Output/{WEAVE_clus}/{xml_2_config}']
    
    for input_xml in fields_to_configure:
        print('Configuring {}'.format(input_xml))
        output_xml = '{}_cfg.xml'.format(os.path.splitext(input_xml)[0])
        config_command = '{} --gui=0 -f {} -o {}'.format(os.path.join(path_to_configure_tools, 'configure'), input_xml, output_xml)
    
        configure = subprocess.run([config_command], shell=True)

def findfile(input_folder, FILE):
    """reads out the positions of configured objects, lifted from obviewer.py. Needs to be ran in the same directory as obviewer.py, protofielding.py etc
    
    Parameters:
        input_folder is where the configured xml is stored
        FILE is the name of the xml file
    
    Returns:
        Configured RA, DEC, TARGPRIO, ID
    """
    def get_xml_properties(path):
        xml = MOSOBXML(path)

        trimester = xml.observation.getAttribute('trimester')
        progtemp = xml.observation.getAttribute('progtemp')
        obstemp = xml.observation.getAttribute('obstemp')
        cfg_ver = xml.configure.getAttribute('configure_version')
        plate = xml.configure.getAttribute('plate')

        return trimester, progtemp, obstemp, cfg_ver, plate

    ob_paths = glob.glob(f'{input_folder}/{FILE}')
    ob_paths.sort()

    ob_cats = []
    ob_names = []
    ob_properties = []

    for path in ob_paths[:]:
        ob = MOSTrackingCat(path, exclude_sky=False)
        ob_names.append(os.path.split(path)[1])

        combined = pd.concat([ob.configured.to_pandas(),
                              ob.unconfigured.to_pandas()],
                              keys=["Configured", "Unconfigured"])
        combined.index.names = ["OB", "i"]

        ob_cats.append(combined)
        ob_properties.append(get_xml_properties(path))

    ob_properties = np.array(ob_properties)
    ob_properties_uniq = [list(np.unique(row)) for row in ob_properties.T]

    data = pd.concat(ob_cats, keys=ob_names)
    
    fibre_status = Select(title="Fibre Status:",
                          options=["Configured", "Unconfigured"],
                          value="Configured")
    positions = Select(title="Coordinate Frame:", options=["Sky", "Plate"],
                       value="Sky")
    ob_select = Select(title="OB to plot:", options=["All"]+ob_names, value="All")

    targ_select = Select(title="Split targets by:",
                         options=["TARGSRVY", "TARGPROG", "TARGPRIO", "TARGCLASS"],
                         value="TARGSRVY")

    def selected():
        if positions.value == "Sky":
            xcol, ycol = "TARGRA", "TARGDEC"

        elif positions.value == "Plate":
            xcol, ycol = "TARGX", "TARGY"

        data["x"] = data[xcol]
        data["y"] = data[ycol]
        data["z"] = data[targ_select.value].astype({targ_select.value: str})

        if ob_select.value == "All":
            selected = data.loc[slice(None), fibre_status.value, slice(None)]
        else:
            selected = data.loc[ob_select.value, fibre_status.value, slice(None)]
        return selected
    f = selected()
    return list(f['TARGRA']), list(f['TARGDEC']), list(f['TARGPRIO']), list(f['TARGNAME'])


def change_prio(prev_fits, out_xml, SaveName, high_prio, low_prio, updated_high_prio, updated_low_prio):
    """
    Updates the priority of configured targets. Note that you will need to change the TARGPRIO's in the first for loop
    
    Parameters:
        prev_fits: fits file that was configured
        out_xml: configured xml
        SaveName: name of fits file that we are saving to
        high_prio: high target priority value
        low_prio: low target priortity value
        updated_high_prio: new priority if high priority target assigned fiber
        updated_low_prio: new priority if low priority target assigned fiber
        
    Returns:
        new fits file with updated priorities
    """
    
    configured_objects = findfile(configured_folder_path, out_xml)
    class Object:

        def __init__(self, x, Names):

            for d,N in zip(x, Names):

                setattr(self, N, d)


    hdul = fits.open(prev_fits)
    Names = (hdul[1].columns).names
    Data = hdul[1].data

    Objects = [Object(x,Names) for x in Data]

    SecondaryHeader = hdul[1].header
    PrimaryHeader = hdul[0].header

    cols = hdul[1].columns

    Names  = cols.names
    Format = cols.formats
    Unit = cols.units
    Null = cols.nulls
    Disp = cols.disps

    ### updates priority
    for x in Objects:
        if x.TARGNAME in configured_objects[3] and x.TARGPRIO == low_prio:
            #print("B")
            x.TARGPRIO = updated_low_prio
        if x.TARGNAME in configured_objects[3] and x.TARGPRIO == high_prio:
            #print("A")
            x.TARGPRIO = updated_high_prio

    Columns = []

    for i, N in enumerate(Names):

        Columns.append(fits.Column(name = N, array = np.array([getattr(x,N) for x in Objects]), format = Format[i], null = Null[i], unit = Unit[i],disp=Disp[i]))

    NEW = fits.BinTableHDU.from_columns(Columns)

    NEWhdr = NEW.header

    for h in SecondaryHeader:
        if h == "HISTORY":
            pass
        else:
            NEWhdr[h] = SecondaryHeader[h]
            NEWhdr.comments[h] = SecondaryHeader.comments[h]

    NEW.writeto(SaveName, overwrite=True)

    NEWhdul = fits.open(SaveName)

    NEWhdr = hdul[0].header

    for h in PrimaryHeader:
        if h == "HISTORY" or h == "COMMENT":
            pass
        else:
            NEWhdr[h] = PrimaryHeader[h]
            NEWhdr.comments[h] = PrimaryHeader.comments[h]
    hdul.close()
    NEWhdul.writeto(SaveName, overwrite = True)

########################------------------- below should be changed depending on cluster -----------------#####################

WEAVE_clus = 'A2626' ###name of cluster
no_tiles = 8 ###number of pointings
clus_file = 'WWFCSA2626FitsLowAs2' ###name of fits file without _1.fits suffix
configured_folder_path = f'/home/ppxdc1/workflow_23/July23/mos/examples/Chaining/Output/{WEAVE_clus}/' ###path to where xml's are stored
hprio, lprio, newhprio, newlprio = 9,2,2,1 ###target priorities
field_prefix = 'WS_2023B1_022' ###field name prefix

for i in range(1,no_tiles+1):
    if i // 2 != i/2:
        PLATE = 'PLATE_A'
        field_file = f'{field_prefix}_{WEAVE_clus}_{i}_MOS_plate_A' ###generic name of field file 
    if i //2 == i/2:
        PLATE = 'PLATE_B'
        field_file = f'{field_prefix}_{WEAVE_clus}_{i}_MOS_plate_B' ###generic name of field file 
        
    first = config(f'{WEAVE_clus}/{clus_file}_{i}.fits', f'{WEAVE_clus}/{field_file}.csv', f'{clus_file}_{i}-{field_file}_01.xml', f'{WEAVE_clus}', PLATE) ###configures
    print(f"configured tile {i}")
    chprio = change_prio(f'input/fits_files/{WEAVE_clus}/{clus_file}_{i}.fits', f'{clus_file}_{i}-{field_file}_01_cfg.xml', f'input/fits_files/{WEAVE_clus}/{clus_file}_{i+1}.fits',hprio, lprio, newhprio, newlprio) ###updates priority
    print(f"Process finished for tile {i}")


