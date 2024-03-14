import numpy as np
import pandas as pd
import glob
import warnings
import os
from os.path import dirname, join

from bokeh.io import output_notebook, show, curdoc, save
from bokeh.layouts import gridplot, layout, column, row
from bokeh.palettes import Reds4, Spectral4, Blues9, viridis
from bokeh.plotting import figure, output_file
from bokeh.models import ColumnDataSource, Div, Select, RadioButtonGroup
from bokeh.transform import factor_cmap, factor_mark

warnings.simplefilter('ignore', category=RuntimeWarning)

import sys
input_folder = sys.argv[1]
#code_path = sys.argv[1] # Path to MOS Workflow Scripts
# Somewhat hacky - to be fixed
#/Users/duncan/Astro/WEAVE/WEAVE-SWG/mos/
#sys.path.append(code_path)
from protofielding import MOSDriverCat, MOSTrackingCat, MOSOBXML

def get_xml_properties(path):
    xml = MOSOBXML(path)

    trimester = xml.observation.getAttribute('trimester')
    progtemp = xml.observation.getAttribute('progtemp')
    obstemp = xml.observation.getAttribute('obstemp')
    cfg_ver = xml.configure.getAttribute('configure_version')
    plate = xml.configure.getAttribute('plate')

    return trimester, progtemp, obstemp, cfg_ver, plate

ob_paths = glob.glob(f'{input_folder}/*.xml')
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
data.index.names = ["OB", "FibStatus", "i"]

dashboard_header = """
<style>
h1 {
    margin: 0.3em 0 0 0;
    color: #2e484c;
    font-family: 'Julius Sans One', sans-serif;
    font-size: 1.8em;
    text-transform: uppercase;
}

h2 {
    margin: 0.2em 0 0 0;
    color: #2e484c;
    font-family: 'Julius Sans One', sans-serif;
    font-size: 1.4em;
}

h3 {
    margin: 0.1em 0 0 0;
    color: #2e484c;
    font-family: 'Julius Sans One', sans-serif;
    font-size: 1.1em;
}

a:link {
    font-weight: bold;
    text-decoration: none;
    color: #0d8ba1;
}
a:visited {
    font-weight: bold;
    text-decoration: none;
    color: #1a5952;
}
a:hover, a:focus, a:active {
    text-decoration: underline;
    color: #9685BA;
}
p {
    font: "Libre Baskerville", sans-serif;
    text-align: left;
    text-justify: inter-word;
    width: 90%;
    max-width: 900;
}

</style>\n
<h1>An Interactive Explorer for WEAVE MOS OBs</h1>
<p>
Interact with the widgets on the right to change which OBs, fibre sets and coordinate frames are shown.
Plotting symbols for individual TARGUSE types can be switched off and on using the corresponding legend entry (NB: Subsets of science targets cannot be individually toggled). Hover over individual sources for additional target information.
</p>\n"""

meta_data_text1 = f"""<p>Plotting {len(ob_paths)} XML files from: <i>{os.path.abspath(input_folder)}</i>
                         (Configure version: {",".join(ob_properties_uniq[3])})
                      </p>\n"""
meta_data_text2 = f"""<p><b>Trimesters</b>: {",".join(ob_properties_uniq[0])} -
                         <b>PROGTEMPs</b>: {",".join(ob_properties_uniq[1])} -
                         <b>OBSTEMPs</b>: {",".join(ob_properties_uniq[2])} -
                         <b>Plates</b>: {",".join(ob_properties_uniq[4])}
                      </p>\n"""

desc = Div(text=dashboard_header+meta_data_text1+meta_data_text2,
           sizing_mode="stretch_width")

"""
Main GUI Plotting Selectors and CDS Setup
"""
fibre_status = Select(title="Fibre Status:",
                      options=["Configured", "Unconfigured"],
                      value="Configured")
positions = Select(title="Coordinate Frame:", options=["Sky", "Plate"],
                   value="Sky")
ob_select = Select(title="OB to plot:", options=["All"]+ob_names, value="All")

targ_select = Select(title="Split targets by:",
                     options=["TARGSRVY", "TARGPROG", "TARGPRIO", "TARGCLASS"],
                     value="TARGSRVY")

targ_values = []
nmax_tvalues = []
palette = ()
for v in targ_select.options:
    vuniq = np.array(list(np.sort(np.unique(data.loc[(data["TARGUSE"] == "T"), (v)]))), dtype='str')
    vuniq = vuniq[vuniq != '']

    targ_values += list(vuniq)
    palette += viridis(len(vuniq))
    nmax_tvalues.append(len(vuniq))

factors = list(targ_values)

# palette=viridis(len(targ_values)),
# factors=list(targ_values))
# targ_values = np.concatenate(unique_tvalues)
# targ_values = targ_values[targ_values != ''] # Science target names for later

tuse_list = ["T", "G", "C", "S"] # Possible TUSE values
labels = ["Targets", "Guide" , "Calib", "Sky"] # Corresponding labels

sources = [ColumnDataSource(data=dict()) for t in labels]

"""
Histogram GUI Selector and CDS Setup
"""
hist_allowed = data.dtypes != ('O')
hist_col = np.array(data.columns.to_list())[hist_allowed]
hist_select = Select(title="Histogram Column:", options=list(hist_col),
                     value="TARGPRIO")

#hist_linlog = RadioButtonGroup(title=)
hist_normscale = RadioButtonGroup(labels=["Total", "Normalised"],
                                  active=0)

# targ_colors = Blues[int(np.clip(len(targ_groups), 3, 9))]

hist, edges = np.histogram(data[hist_select.value], bins=10)
hist_sources = [ColumnDataSource(data=dict({'hist':hist,
                                            'left':edges[:-1],
                                            'right':edges[1:]})) for t in labels]


colors = Reds4

# targ_groups = dict(list(np.unique(source.data["z"]))

tooltips = [ # Label : @Dataframe column-name pairs
    ("CNAME", "@CNAME"),
    ("ID", "@TARGID"),
    ("NAME", "@TARGNAME"),
    ("SRVY", "@TARGSRVY"),
    ("PROG", "@TARGPROG"),
    ("PRIO", "@TARGPRIO"),
    ("(x,y)", "($x, $y)"),
]

"""
Set up plotting windows and scatter/histogram plots
"""
plot_title = Div(text='<h2 style="text-align: center">Fibre Distribution</h2>',
                 sizing_mode="stretch_width")
p = figure(width=750, aspect_ratio = 1.0, tooltips=tooltips,
           toolbar_location="above")
h = figure(sizing_mode="scale_width", aspect_ratio = 1.2, toolbar_location="right")
h.yaxis.axis_label = "N"

for i, source in enumerate(sources):
    if i==0:
        p.scatter(x="x",
                  y="y",
                  source=source,
                  legend_field = "z",
                  color=factor_cmap("z",
                                    palette=palette,
                                    factors=factors),
                  marker='circle',
                  size=5, alpha=0.8)


        h.quad(top='hist', bottom=0, left='left', right='right',
               source=hist_sources[i],
               fill_color=Blues9[0],
               fill_alpha=0.5,
               line_color=Blues9[0],
               legend_label=labels[i])

    else:
        color = colors[i-1]

        p.circle(x="x",
                 y="y",
                 source=source, color=color,
                 size=5, alpha=0.8, legend_label = labels[i])

        h.quad(top='hist', bottom=0, left='left', right='right',
               source=hist_sources[i],
               line_color=colors[i-1],
               fill_alpha=0.5,
               fill_color=colors[i-1], legend_label=labels[i])

"""
Define update/selection functions for interactive updates
"""
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

def update():
    df = selected()

    # Update x & y inputs and labels
    if positions.value == "Sky":
        xcol, ycol = "TARGRA", "TARGDEC"
        p.xaxis.axis_label = 'RA'
        p.yaxis.axis_label = 'Dec'

    elif positions.value == "Plate":
        xcol, ycol = "TARGX", "TARGY"
        p.xaxis.axis_label = 'X'
        p.yaxis.axis_label = 'Y'

    for i, (source, hsource, tuse) in enumerate(zip(sources,
                                                hist_sources,
                                                tuse_list)):

        subdf = df.loc[df["TARGUSE"] == tuse]
        source.data = subdf # Fill source with sliced dataframe

        if hist_normscale.active == 0:
            hdensity = False
        elif hist_normscale.active == 1:
            hdensity = True

        if tuse == "T": # Set histogram bins based on target properties
            unique = np.unique(subdf[hist_select.value])
            if len(unique) <= 10 and not np.isnan(unique).all():
                nbins = np.max(unique)+1
                range = (-0.5, np.max(unique)+0.5)
                hist, edges = np.histogram(subdf[hist_select.value], bins=int(nbins),
                                           range=range, density=hdensity)

            elif len(unique) > 10 and not np.isnan(unique).all():
                use = np.isfinite(subdf[hist_select.value])
                range = np.percentile(subdf[hist_select.value][use], [0,100])
                hist, edges = np.histogram(subdf[hist_select.value], bins='auto',
                                           density=hdensity, range=np.round(range, 1))

            else:
                hist, edges = np.histogram(subdf[hist_select.value],
                                           range=(0,1), bins=10, density=hdensity)

        else: # Use target binedges for G/C/S targets
            hist, edges = np.histogram(subdf[hist_select.value], bins=edges,
                                       density=hdensity)

        hsource.data = {'hist': hist, 'left': edges[:-1], 'right':edges[1:]}
    h.xaxis.axis_label = hist_select.value

# Scatter plot legend properties
p.legend.location = "top_left"
p.legend.click_policy="hide"
p.legend.label_height = 12
p.legend.glyph_height = 12

# Histogram legend properties
h.legend.location = "top_left"
h.legend.click_policy="hide"
h.legend.background_fill_alpha = 0.1
h.legend.glyph_width = 12
h.legend.glyph_height = 12
h.legend.label_height = 12


"""
Dashboard layout and stucture
"""
col_title1 = Div(text='<h3 style="text-align: center">Global Properties</h3>',
                 sizing_mode="stretch_width")
col_title2 = Div(text='<h3 style="text-align: center">Scatter Plot Properties</h3>',
                sizing_mode="stretch_width")
controls = [ob_select, fibre_status, positions, targ_select]

hist_title = Div(text='<h3 style="text-align: center">Histogram Properties</h3>',
                sizing_mode="stretch_width")
hist_boxes = [hist_title, row(hist_select, hist_normscale), h]

for control in [*controls, hist_select]: # Link controls to scatter plot update
   control.on_change('value', lambda attr, old, new: update())

for control in [hist_normscale]: # Link radio button controls to update
    control.on_change('active', lambda attr, old, new: update())


inputs = column(col_title1, row(*controls[:2]), col_title2, row(*controls[2:]), *hist_boxes, width=600)
pcol = column(plot_title, p, sizing_mode="stretch_both")
l = column(desc, row(pcol, inputs))

update()  # initial load of the data
curdoc().add_root(l)
