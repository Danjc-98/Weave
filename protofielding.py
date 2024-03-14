#!/usr/bin/env python3

#
# Copyright (C) 2021 - Kenneth Duncan
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <https://www.gnu.org/licenses/>.
#

import argparse
import glob
import logging
import os
import re
import time
import warnings
import xml.dom.minidom
from collections import OrderedDict

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack, join
from astropy.coordinates import SkyCoord, FK5, ICRS
import astropy.units as u
from astropy.time import Time
from astropy.wcs import WCS, utils
from astropy.stats import mad_std
from sklearn.utils import resample

# Import levels to be amended when ready to integrate into the WEAVE bitbucket
from workflow.utils.classes import OBXML
from workflow.utils.classes._guide_and_calib_stars import GuideStars as _GuideStars
from workflow.utils.classes._guide_and_calib_stars import CalibStars as _CalibStars

from workflow.utils import get_version
from workflow.utils.get_obstemp_info import (get_obstemp_info,
                                             get_obstemp_dict)
from workflow.utils.get_progtemp_info import (get_progtemp_dict,
                                              get_progtemp_info,
                                              get_obsmode_from_progtemp,
                                              get_resolution_from_progtemp)
from workflow.utils.get_resources import (get_blank_xml_template,
                                          get_progtemp_file,
                                          get_obstemp_file)


from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)


class MOSOBXML(OBXML):
    """ Modified OBXML Class for WEAVE MOS XML creation

    This class builds upon the IFU workflow OBXML class, adds new functionality
    necessary or useful for MOS XML creation and analysis. Unless specifically
    overwritten by the new attributes below, all functionality is inherited from
    the base OBXML class.
    """
    def set_fields(self, field_ra, field_dec):
        # A dictionary for mapping column names of the entries to attributes
        col_to_attrib_dict = {
            'TARGSRVY': 'targsrvy',
            'TARGPROG': 'targprog',
            'TARGID': 'targid',
            'TARGNAME': 'targname',
            'TARGPRIO': 'targprio',
            'GAIA_RA': 'targra',
            'GAIA_DEC': 'targdec',
            'GAIA_EPOCH': 'targepoch',
            'GAIA_PMRA': 'targpmra',
            'GAIA_PMDEC': 'targpmdec',
            'GAIA_PARAL': 'targparal'
        }

        # Get a empty field element to use it as template

        field_list = self.fields.getElementsByTagName('field')

        field_template = field_list[0].cloneNode(True)

        for node in field_template.getElementsByTagName('target'):
            field_template.removeChild(node)

        for node in field_template.childNodes:
            if node.nodeType is xml.dom.minidom.Node.COMMENT_NODE:
                field_template.removeChild(node)

        # Get a target element and save it as template
        target_list = self.fields.getElementsByTagName('target')
        target_template = None

        for target in target_list:
            if target.getAttribute('targuse') == 'T':
                target_template = target.cloneNode(True)
                break

        assert target_template is not None

        # Clean the fields element
        for node in self.fields.getElementsByTagName('field'):
            self.fields.removeChild(node)
        order = "" # self.first_science_order

        field = field_template.cloneNode(True)

        field_attrib_dict = {
            'Dec_d': field_dec,
            'RA_d': field_ra,
            'order': order
            }

        self._set_attribs(field, field_attrib_dict)
        self.central_ra = field_ra
        self.central_dec = field_dec
        # Add a target per entry to the field

        #for entry in entry_group:

        target = target_template.cloneNode(True)

        # target_attrib_dict = {}
        #
        # target_attrib_dict['targuse'] = 'T'
        # target_attrib_dict['targclass'] = 'UNKNOWN'
        # target_attrib_dict['targcat'] = targcat
        #
        # for col in col_to_attrib_dict.keys():
        #     target_attrib_dict[col_to_attrib_dict[col]] = entry[col]
        #
        # self._set_attribs(target, target_attrib_dict)

        field.appendChild(target)

        # Add the field to fields

        self.fields.appendChild(field)

    def remove_configure_outputs(self):

        # Remove some attributes from the observation element

        configure_attrib_list = ['configure_version', 'seed']

        for attrib in configure_attrib_list:
            if attrib in self.configure.attributes.keys():
                self.configure.removeAttribute(attrib)

        # Remove some attributes from the target elements

        target_attrib_list = [
            'automatic', 'configid', 'fibreid', 'ifu_pa', 'ifu_spaxel',
            'targx', 'targy'
        ]

        for node in self.dom.getElementsByTagName('target'):
            for attrib in target_attrib_list:
                if attrib in node.attributes.keys():
                    node.removeAttribute(attrib)

        # Remove some elements

        elem_list = [
            'optical_axis', 'distortion_coefficients',
            'telescope', 'focal_plane_map', 'hour_angle_limits'
        ]

        self._remove_elem_list_by_name(elem_list)

    def clean_target_list(self):

        num_removed_targets = 0

        for field in self.fields.getElementsByTagName('field'):
            for target in field.getElementsByTagName('target'):
                if str(target.getAttribute('targcat')) == '%%%':
                    num_removed_targets += 1
                    field.removeChild(target)
                else:
                    target.setAttribute('ifu_spaxel', '')
                    target.setAttribute('ifu_pa', '')
                    target.setAttribute('automatic', '0')

                if str(target.getAttribute('targuse')) == 'S':
                    for photometry in target.getElementsByTagName('photometry'):
                        target.removeChild(photometry)

        if num_removed_targets > 0:
            logging.info('{} non-allocated targets removed'.format(
                         num_removed_targets))

    def _set_epoch(self, epoch):
        self.conditions = self.dom.getElementsByTagName('conditions')[0]

        conditions_dict = {'epoch': '{:.2f}'.format(epoch.byear)}
        self._set_attribs(self.conditions, conditions_dict)

    def _get_guide_stars(self, plot_filename=None, useful_table_filename=None,
                         mos_num_stars_request=8, mos_num_central_stars=1,
                         mos_min_cut=0.8, mos_max_cut=0.95, dyn_range_lim=2.0,
                         cache=False, mag_lim=99.):

        obsmode = self._get_obsmode()

        pa = self._get_pa()
        max_guide = self._get_max_guide()
        resolution = get_resolution_from_progtemp(self._get_progtemp())

        guide_stars = _GuideStars(self.central_ra, self.central_dec, obsmode, resolution)

        actual_pa, guides_table = guide_stars.get_table(
            verbose=True, plot_filename=plot_filename,
            useful_table_filename=useful_table_filename, pa_request=pa,
            num_stars_request=None,
            num_central_stars=None,
            min_cut=0.0, max_cut=0.95,
            cache=cache, mag_lim=mag_lim)

        # Trim full table to MOS suitable candidates
        useful_table = guides_table[guides_table['TARGPROG'] == 'ANY']

        # Do central / annulus filtering with revised list
        central_star_table, non_selected_table = (
                guide_stars._select_central_stars_in_plate_a_b(
                useful_table, num_central_stars=mos_num_central_stars))

        if mos_num_stars_request is not None:
            mos_num_ring_stars = mos_num_stars_request - mos_num_central_stars
        else:
            mos_num_ring_stars = None

        # Filter ring stars based to limit dynamic range
        bright_lim = np.min(central_star_table['GAIA_MAG_G'])
        gmag_diff = np.abs(non_selected_table['GAIA_MAG_G'] - bright_lim)
        non_selected_table = non_selected_table[gmag_diff < dyn_range_lim]

        ring_stars_table = guide_stars._select_stars_in_ring_of_plate_a_b(
            non_selected_table, len(non_selected_table),
            min_cut=mos_min_cut, max_cut=mos_max_cut)

        ring_idx = resample(np.arange(len(ring_stars_table)), 
                    n_samples=mos_num_ring_stars, 
                    replace=False, 
                    stratify=((ring_stars_table['ANGLE']+180) / 45).astype('int'))

        selected_table = vstack([central_star_table, ring_stars_table[ring_idx]])

        return actual_pa, selected_table

    def _add_guide_stars(self, plot_filename=None, useful_table_filename=None,
                         mos_num_stars_request=8, mos_num_central_stars=1,
                         mos_min_cut=0.8, mos_max_cut=1.0, dyn_range_lim=2.0,
                         cache=False, mag_lim=99.):

        actual_pa, guides_table = self._get_guide_stars(
            plot_filename=plot_filename,
            useful_table_filename=useful_table_filename,
            mos_num_stars_request=mos_num_stars_request,
            mos_num_central_stars=mos_num_central_stars,
            mos_min_cut=mos_min_cut, mos_max_cut=mos_max_cut,
            dyn_range_lim=dyn_range_lim, cache=cache,
            mag_lim=mag_lim)

        self._set_guide_stars(actual_pa, guides_table)

    def _get_calib_stars(self, resolution, plot_filename=None,
                         useful_table_filename=None,
                         num_stars_request=20, num_central_stars=0,
                         min_cut=0.0, max_cut=1.0, 
                         cache=False, mag_lim=99., calib_mixres=False):

        obsmode = self._get_obsmode()
        resolution = get_resolution_from_progtemp(self._get_progtemp())

        calib_stars = _CalibStars(self.central_ra, self.central_dec, obsmode, resolution)

        calibs_table = calib_stars.get_table(
            verbose=True, plot_filename=plot_filename,
            useful_table_filename=useful_table_filename,
            num_stars_request=num_stars_request,
            num_central_stars=num_central_stars,
            min_cut=min_cut, max_cut=max_cut,
            cache=cache, mag_lim=mag_lim, calib_mixres=calib_mixres)


        # Do central / annulus filtering with revised list
        central_star_table, non_selected_table = (
                calib_stars._select_central_stars_in_plate_a_b(
                calibs_table, num_central_stars=num_central_stars))

        if num_stars_request is not None:
            num_ring_stars = num_stars_request - num_central_stars
        else:
            num_ring_stars = None

        ring_stars_table = calib_stars._select_stars_in_ring_of_plate_a_b(
            non_selected_table, num_ring_stars,
            min_cut=min_cut, max_cut=max_cut)

        selected_table = vstack([central_star_table, ring_stars_table])

        return selected_table

    def _add_calib_stars(self, resolution,
                         plot_filename=None, useful_table_filename=None,
                         num_stars_request=2, num_central_stars=0, min_cut=0.2,
                         max_cut=0.4, add_cnames=True, cache=False,
                         mag_lim=99., calib_mixres=False):

        obsmode = self._get_obsmode()

        if obsmode == 'LIFU':
            # No calibration stars needed!
            return

        calibs_table = self._get_calib_stars(resolution,
                                             plot_filename=plot_filename,
                                             useful_table_filename=useful_table_filename,
                                             num_stars_request=num_stars_request,
                                             num_central_stars=num_central_stars,
                                             min_cut=min_cut, max_cut=max_cut,
                                             cache=cache, mag_lim=mag_lim, 
                                             calib_mixres=calib_mixres)

        self._set_calib_stars(calibs_table, add_cnames=add_cnames)

    def add_guide_and_calib_stars(self, resolution,
                                  guide_plot_filename=None,
                                  guide_useful_table_filename=None,
                                  mos_num_guide_stars_request=8,
                                  mos_num_central_guide_stars=1,
                                  mos_min_guide_cut=0.8,
                                  mos_max_guide_cut=1.0,
                                  dyn_range_lim=2.0,
                                  calib_plot_filename=None,
                                  calib_useful_table_filename=None,
                                  num_calib_stars_request=20,
                                  num_central_calib_stars=0,
                                  min_calib_cut=0.0,
                                  max_calib_cut=1.0,
                                  calib_mixres=False,
                                  cache=False,
                                  guide_mag_lim=99.,
                                  calib_mag_lim=99.):

        self._add_guide_stars(plot_filename=guide_plot_filename,
                              useful_table_filename=guide_useful_table_filename,
                              mos_num_stars_request=mos_num_guide_stars_request,
                              mos_num_central_stars=mos_num_central_guide_stars,
                              mos_min_cut=mos_min_guide_cut,
                              mos_max_cut=mos_max_guide_cut,
                              dyn_range_lim=dyn_range_lim,
                              cache=cache,
                              mag_lim=guide_mag_lim)

        self._add_calib_stars(resolution=resolution,
                              plot_filename=calib_plot_filename,
                              useful_table_filename=calib_useful_table_filename,
                              num_stars_request=num_calib_stars_request,
                              num_central_stars=num_central_calib_stars,
                              min_cut=min_calib_cut,
                              max_cut=max_calib_cut,
                              calib_mixres=calib_mixres,
                              cache=cache,
                              mag_lim=calib_mag_lim)


    def target_statistics(self, attribute, use_all=False):
        """
        Construct a dictionary with target level statistics

        Parameters
        ----------
        attribute : str
            Configure target attribute for which to calculate statistics
        use_all : bool, default=False
            Include non science targets (i.e. guide, calibration, sky) when
            calculating statistics
        """
        attr_dict = {}
        for field in self.fields.getElementsByTagName('field'):
            for target in field.getElementsByTagName('target'):
                tuse = str(target.getAttribute('targuse'))

                if (tuse == 'T') or use_all:
                    tvalue = str(target.getAttribute(attribute.lower()))
                    if tvalue in attr_dict.keys():
                        attr_dict[tvalue] += 1
                    else:
                        attr_dict[tvalue] = 1
        return attr_dict


    def summary(self, verbose=True):
        """
        Calculate (and print) summary statistics for the current OB XML

        Parameters
        ----------
        verbose : bool, default=True
            Print summary statistics to the command line.

        """

        self._tuse_configured_dict = {'G': 0, # Pre-generate TUSE dict to present all types
                                      'C': 0,
                                      'T': 0,
                                      'S': 0}

        self._tuse_unconfigured_dict = {'G': 0, # Pre-generate TUSE dict to present all types
                                        'C': 0,
                                        'T': 0,
                                        'S': 0}

        for field in self.fields.getElementsByTagName('field'):
            for target in field.getElementsByTagName('target'):
                tuse = str(target.getAttribute('targuse'))
                if 'fibreid' in target.attributes.keys():
                    self._tuse_configured_dict[tuse] += 1
                else:
                    self._tuse_unconfigured_dict[tuse] += 1

        self._srvy_dict = self.target_statistics('targsrvy')
        self._prog_dict = self.target_statistics('targprog')

        if verbose:
            print('Configured TARGUSE Statistics:')
            for tuse, count in self._tuse_configured_dict.items():
                print(f'{tuse} {count:>5}')

            print('\nUnconfigured TARGUSE Statistics:')
            for tuse, count in self._tuse_unconfigured_dict.items():
                print(f'{tuse} {count:>5}')

            print('\nTARGSRVY Statistics:')
            for srvy, count in self._srvy_dict.items():
                print(f'{srvy:<15} {count:>5}')

            print('\nTARGPROG Statistics:')
            for prog, count in self._prog_dict.items():
                print(f'{prog:<15} {count:>5}')

    def remove_unconfigured(self, verbose=False):
        removed = 0
        for field in self.fields.getElementsByTagName('field'):
            for target in field.getElementsByTagName('target'):
            # Delete the allocated targets identified by the 'fibre' keyword
                if 'fibreid' not in target.attributes.keys():
                    p = target.parentNode
                    p.removeChild(target)
                    removed += 1
        if verbose:
            print(f'{removed} unconfigured targets removed')

    def remove_configured(self, tracking_fits, filter_dict,
                          unique_col='CNAME', ignore_guide=True,
                          ignore_calib=True, rtol=0, atol=1e-5, verbose=False):

        progtemp = self.observation.getAttribute('progtemp')
        obstemp = self.observation.getAttribute('obstemp')

        # Limit tracking catalog to correct PROGTEMP and OBSTEMP
        parent_mask = (tracking_fits.configured['PROGTEMP'] == progtemp)
        parent_mask *= (tracking_fits.configured['OBSTEMP'] == obstemp)

        removed = 0

        for field in self.fields.getElementsByTagName('field'):
            for target in field.getElementsByTagName('target'):
                targuse = target.getAttribute('targuse')

                if targuse == 'G' and ignore_guide:
                    continue
                if targuse == 'C' and ignore_calib:
                    continue

                targ_unique_col = target.getAttribute(unique_col.lower())
                row_match = (tracking_fits.configured[unique_col] == targ_unique_col)
                row_match *= parent_mask

                # Verify that coordinates match
                row_match *= np.isclose(tracking_fits.configured['TARGRA'],
                                        float(target.getAttribute('targra')),
                                        rtol=rtol, atol=atol)
                row_match *= np.isclose(tracking_fits.configured['TARGDEC'],
                                        float(target.getAttribute('targdec')),
                                        rtol=rtol, atol=atol)

                rx = np.flatnonzero(row_match)

                if len(rx) == 1:
                    n_cfg = tracking_fits.configured['NCONFIG'][rx[0]]
                    if n_cfg >= filter_dict[target.getAttribute('targprog')]:
                        p = target.parentNode
                        p.removeChild(target)
                        removed += 1

                # How to deal with multiple matches (i.e. different surveys)
                # still TBD
                elif len(rx) > 1:
                    raise Exception

                else:
                    continue

        if verbose:
            print(f'{removed} previously configured targets removed')

    def update_progtemp(self, progtemp, progtemp_file=None, xml_template=None):
        """ Update the OB PROGTEMP

        Spectrograph attributes are updated in place. Exposure atttributes are
        reset to the blank template before being rebuilt from scratch according
        to the new PROGTEMP.

        Parameters
        ----------
        progtemp : string
            Desired new PROGTEMP for OB.
        progtemp_file : string
            If 'None',

        """
        if progtemp_file == None:
            progtemp_file = 'progtemp.dat'
            get_progtemp_file(progtemp_file)

        self.progtemp_datamver, self.progtemp_dict, self.forbidden_dict = get_progtemp_dict(
            filename=progtemp_file, assert_orb=True)

        # Guess some parameters which depends on PROGTEMP
        chained = ('+' in progtemp)

        spectrograph_dict = get_progtemp_info(progtemp,
                                              progtemp_dict=self.progtemp_dict)

        obsmode = spectrograph_dict['obsmode']

        assert obsmode in ['MOS']

        assert (spectrograph_dict['red_resolution'] ==
                spectrograph_dict['blue_resolution'])
        resolution = spectrograph_dict['red_resolution']

        red_vph = spectrograph_dict['red_vph']
        blue_vph = spectrograph_dict['blue_vph']

        assert (spectrograph_dict['red_num_exposures'] ==
                spectrograph_dict['blue_num_exposures'])
        num_science_exposures = spectrograph_dict['red_num_exposures']

        assert (spectrograph_dict['red_exp_time'] ==
                spectrograph_dict['blue_exp_time'])
        science_exptime = spectrograph_dict['red_exp_time']

        assert (spectrograph_dict['red_binning_x'] ==
                spectrograph_dict['blue_binning_x'])
        binning_x = spectrograph_dict['red_binning_x']

        # Set the spatial binning from the input of this method

        binning_y = 1

        self.set_spectrograph(binning_x=binning_x, binning_y=binning_y,
                                resolution=resolution, red_vph=red_vph,
                                blue_vph=blue_vph)


        if xml_template == None:
            self.xml_template = 'BlankXMLTemplate.xml'
            get_blank_xml_template(self.xml_template)
        else:
            self.xml_template = xml_template

        template = MOSOBXML(self.xml_template)

        for exposure in self.exposures.getElementsByTagName('exposure'):
            self.exposures.removeChild(exposure)

        for exposure in template.exposures.getElementsByTagName('exposure'):
            node = exposure.cloneNode(True)
            self.exposures.appendChild(node)

        # Set the contents of the exposures element

        self.set_exposures(num_science_exposures, science_exptime)

        for exposure in self.exposures.getElementsByTagName('exposure'):
            if exposure.getAttribute('type') == 'science':
                exposure.setAttribute('speed', 'slow')

        trimester = self.observation.getAttribute('trimester')
        observation_name = self.observation.getAttribute('name')
        obsmode = self.observation.getAttribute('obs_mode')
        obstemp = self.observation.getAttribute('obstemp')

        observation_attrib_dict = {
            'chained': chained,
            'name': observation_name,
            'obstemp': obstemp,
            'obs_mode': obsmode,
            'progtemp': progtemp,
            'trimester': trimester
        }

        self.set_observation(observation_attrib_dict)


    def update_obstemp(self, obstemp, obstemp_file=None):
        """ Update the OB OBSTEMP

        obscontraints attributes are updated in place to match the desired new
        obstemp code. Note that no checks are performed to ensure that the
        relevant

        Parameters
        ----------
        obstemp : string
            Desired new OBSTEMP for OB.
        obstemp_file : string
            If 'None',

        """
        if obstemp_file == None:
            obstemp_file = 'obstemp.dat'
            get_obstemp_file(obstemp_file)

        self.obstemp_datamver, self.obstemp_dict = get_obstemp_dict(
            filename=obstemp_file)

        obsconstraints_dict = get_obstemp_info(obstemp,
                                               obstemp_dict=self.obstemp_dict)

        elevation_min = obsconstraints_dict['elevation_min']
        moondist_min = obsconstraints_dict['moondist_min']
        seeing_max = obsconstraints_dict['seeing_max']
        skybright_max = obsconstraints_dict['skybright_max']
        transparency_min = obsconstraints_dict['transparency_min']

        progtemp = self.observation.getAttribute('progtemp')
        chained = ('+' in progtemp)

        trimester = self.observation.getAttribute('trimester')
        observation_name = self.observation.getAttribute('name')
        obsmode = self.observation.getAttribute('obs_mode')

        observation_attrib_dict = {
            'chained': chained,
            'name': observation_name,
            'obstemp': obstemp,
            'obs_mode': obsmode,
            'progtemp': progtemp,
            'trimester': trimester
        }

        self.set_observation(observation_attrib_dict)

        obsconstraints_attrib_dict = {
            'elevation_min': elevation_min,
            'moondist_min': moondist_min,
            'seeing_max': seeing_max,
            'skybright_max': skybright_max,
            'transparency_min': transparency_min
        }

        self.set_obsconstraints(obsconstraints_attrib_dict)


class MOSDriverCat:
    """
    Convert the target-level data from a MOS catalogue to a set of XMLs.

    This class provides code to take a MOS catalogue containing a set of MOS
    targets and produce protofield XMLs at the desired field centres.

    Parameters
    ----------
    filename : str
        A FITS file containing a MOS cat.
    targcat : str, optional
        Filename of the FITS catalogue which will be submitted to WASP (if None,
        it will be derived from the input filename).
    xml_template: str, optional
        Filename of the xml_template to use (if None, will revert to standard
        BlankXMLTemplate)
    """


    def __init__(self, filename, targcat=None, xml_template=None,
                 progtemp_file=None, obstemp_file=None):

        file_list = list(np.array(filename, ndmin=1))

        # Save the filename
        self.filenames = file_list

        # Set the default prefix to be used for output XMLs

        self.default_prefix = os.path.splitext(
            os.path.basename(self.filenames[0]))[0].replace('-mos_driver_cat', '')

        if self.default_prefix[-1] != '-':
            self.default_prefix = self.default_prefix + '-'

        # Set the targcat value
        if targcat is None:
            self.targcat = os.path.basename(self.filenames[0]).replace(
                '-mos_driver_cat', '')
        else:
            self.targcat = targcat

        # Read and save the information into the MOS driver catalogue
        header_keywords = ['DATAMVER', 'TRIMESTE', 'CAT_MAIL',
                           'CAT_CC', 'TACALLOC', 'TACID']

        self.header_values = {}
        for key in header_keywords:
            self.header_values[key] = []

        data = []

        for filename in self.filenames:
            logging.info('Reading {}'.format(filename))
            with fits.open(filename) as hdu_list:
                for key in header_keywords:
                    try:
                        self.header_values[key].append(hdu_list[0].header[key])
                    except:
                        logging.warning(f'"{key}" FITS header keywords not present - submission may fail WASP')
                        self.header_values[key] = ['']

                data.append(Table(hdu_list[1].data))

        self.data = vstack(data) # Merge catalogues

        # Multiple input catalogue consistency checks
        if len(self.filenames) > 1:
            for key in ['TRIMESTE', 'TACALLOC', 'TACID']:
                try:
                    assert (np.array(self.header_values[key]) == self.header_values[key][0]).all()

                except:
                    logging.warning(f'"{key}" FITS Header keywords do not all match: {self.header_values[key]}')


        if xml_template == None:
            self.xml_template = 'BlankXMLTemplate.xml'
            get_blank_xml_template(self.xml_template)
        else:
            self.xml_template = xml_template

        xml_template_dom = xml.dom.minidom.parse(self.xml_template)
        self.xml_datamver = xml_template_dom.childNodes[0].getAttribute('datamver')

        if progtemp_file == None:
            progtemp_file = 'progtemp.dat'
            get_progtemp_file(progtemp_file)

        self.progtemp_datamver, self.progtemp_dict, self.forbidden_dict = get_progtemp_dict(
            filename=progtemp_file, assert_orb=True)

        if obstemp_file == None:
            obstemp_file = 'obstemp.dat'
            get_obstemp_file(obstemp_file)

        self.obstemp_datamver, self.obstemp_dict = get_obstemp_dict(
            filename=obstemp_file)


    def process_ob(self,
                   field_ra, field_dec, field_name,
                   progtemp, obstemp, plate,
                   selection_dict=None,
                   targsrvy_dict=None,
                   avoidance_list=False,
                   epoch=None, report_verbosity=0, radius=1,
                   output_dir='', prefix=None, suffix='',
                   max_sky = None, max_guide = None,
                   num_guide_stars_request=None, dyn_range_lim=3.0,
                   num_central_guide_stars=1,
                   min_guide_cut=0.3, max_guide_cut=0.7,
                   max_calibration = None,
                   num_calib_stars_request = None,
                   num_central_calib_stars = 0,
                   min_calib_cut = 0., max_calib_cut = 1.0,
                   calib_mixres=False, pa = None,
                   linkedgroup = '', linkedgroup_priority = None,
                   obsgroup = '', obsgroup_validity = None, 
                   gaia=True, tycho=True, twomass=True, lga=True,
                   avoidance_maglim=16.0, avoidance_scale=1,
                   cache=False, guide_mag_lim=99., calib_mag_lim=99.):
        """ Generate MOS OB from the driver catalog with the desired properties

        Parameters
        ----------
        field_ra : float [deg]
            Right ascension of the desired protofield centre
        field_dec : float [deg]
            Declination of the desired protofield centre
        field_name : str
            Name assigned to protofield - embedded within the XML data itself
            # TBD - Set up automated default if not manually set?
        progtemp : str
            PROGTEMP of the desired protofield - Spectrograph and exposure level
            information will be set according to the latest PROGTEMP definitions.
            Targets in the driver catalog will be filtered to match the
            requested PROGTEMP value.
        obstemp : str
            OBSTEMP of the desired protofield - Observing constraints will be
            set according to the latest OBSTEMP definitions. Targets in the
            driver catalog will be filtered to match the requested OBSTEMP value.
        plate : {'PLATE_A', 'PLATE_B'}
            Observing plate to use for protofield.
        selection_dict : dict, (optional)
            Selection dictionary used to control targets included in protofield.
            With format: `{'TARGPROG_1': [ COLUMN , N , REVERSE , FIXED_N ], ...}` 
            where COLUMN must be available in the FITS cataloge. N is the number of
            targets (`FIXED_N = TRUE`) or selection threshold (`FIXED_N=FALSE`) 
            associated with this column. If REVERSE=TRUE, targets are selected 
            in ascending order (i.e. for magnitudes).
        targsrvy_dict : dict, (optional)
            Dictionary to set `<survey>` attributes in the output XML. The dict
            must include a `_max_fibres` and `_priority` value for each
            TARGSRVY being included in the output MOS file. If not included,
            each TARGSRVY is set to the maximum available fibres and even 
            priority.
        avoidance_list : bool
            Generate bright target avoidance list for OB. Additional optional
            flags to control the properties of the avoidance targets are 
            included below.
        epoch : float
            Epoch of observations - required by Configure (but can be
            overwritten when run through command line). If 'None', the current
            time is converted to a valid epoch.
        report_verbosity : int {0, 1}
            WASP Submission report verbosity flag.
        radius : float [deg], default=1
            Radius of FoV to use when selecting targets from catalog. Default to
            1 deg radius - but can be extended in case field centre variations
            may be explored (to optimise fibre assignment).
        output_dir : str
            Output directory to save output XML
        prefix : str
            Output filename prefix. If 'None', output XML prefix is set to the
            driver catalog name. XMLs are saved to:
            '{output_dir}/{prefix}-{field_name}-{N}-{suffix}.xml'
            where N is the first index which does not exist for a given filename
            in the referenced output folder.
        suffix : str, optional
            Filename suffix to differentiate protofield stages, e.g. 'tgc' would
            match IFU workflow convention. See above for full filename.
        max_sky : int, optional
            Maximum number of sky fibres in configure. Default set by
            BlankXMLTemplate, non-standard values may not pass WASP check.
        max_guide : int, optional
            Maximum guide star attribute for Configure. Default set by
            BlankXMLTemplate, non-standard values may not pass WASP check.
        num_guide_stars_request : int, optional
            Maximum number of guide stars in the output. None means no limit.
        num_central_guide_stars : int, optional
            Number of stars near to centre to be selected.
        min_guide_cut : float, optional
            Minimum annuli radius to be used for the non-central guide stars.
        max_guide_cut : float, optional
            Maximum annuli radius to be used for the non-central guide stars.
        max_calibration : int, optional
            Maximum calibration star attribute for Configure. Default set by
            BlankXMLTemplate, non-standard values may not pass WASP check.
        num_calib_stars_request : int, optional
            Maximum number of calibration stars in the output. None means no
            limit.
        num_calib_central_stars : int, optional
            Number of calibration stars near to centre to be selected.
        min_calib_cut : float, optional
            Minimum annuli radius to be used for the non-central calibration
            stars.
        max_calib_cut : float, optional
            Maximum annuli radius to be used for the non-central calibration
            stars.
        calib_mixres : bool,
            Allow for mixed resolution calibration stars for HR observations
        pa : float, optional
            Position angle for WEAVE field. For MOS, this is left blank for
            configure to determine the optimal PA for the OB. Only fix with
            good reason to avoid issues with Guide/Calibration assignment.
        linkedgroup : str, optional
            Linking pointer for intra-survey grouping - OBs with the same
            attribute value will be linked together. Setting this value may
            impact scheduling operations - .
        linkedgroup_priority: str, optional [Not yet implemented]
             Intra-survey priority of linked groups with respect to each other.
             Allows preferential selection (and commencement) of a particular
             linked-group as a priority. [Not yet implemented]
        obsgroup : str, optional
            Linking pointer for OB groupings that can permit prioritisation over
            other surveys.
        obsgrous_validity : int,
            Lifetime (days) of obsgroup OB priority boost once 1st OB is 
            observed.
        gaia : bool, (default=True)
            Include bright Gaia stars in avoidance list
        tycho : bool, (default = True)
            Include Tycho stars in avoidance list
        twomass : bool, (default = True)
            Include 2MASS Extended Source Catalogue galaxies within avoidance
            list
        lga : bool, (default = True)
            Include 2MASS Large Galaxy Atlas sources within the avoidance list
        avoidance_maglim : float, (default = 16)
            Faintest magnitude (Gaia G or 2MASS K) to include in avoidance list
        avoidance_scale : float, (default = 1)
            Normalisation of the magnitude dependent avoidance radius. Can be
            used to adjust masks based on outputs from configure.
        cache : bool, optional
             Do not download catalogue files if they aready exist
        guide_mag_lim : float, optional
             Apply a faint limit cut in Gaia G to the available guide stars        
        calib_mag_lim : float, optional
             Apply a faint limit cut in Gaia G to the available calibration stars        
        """

        field_centre = SkyCoord(field_ra*u.degree, field_dec*u.degree, frame='icrs')
        print("Field centre: {0:.3f} {1:.3f} (deg)".format(field_centre.ra.deg,
                                                           field_centre.dec.deg))
        ralist = self.data['GAIA_RA']
        declist = self.data['GAIA_DEC']
        catalog = SkyCoord(ralist*u.degree, declist*u.degree, frame='icrs')
        self.catmask = field_centre.separation(catalog) < radius*u.deg
        self.prog_obs_cut = ((self.data['PROGTEMP'] == progtemp) *
                             (self.data['OBSTEMP'] == obstemp))

        self.report_verbosity = int(np.clip(report_verbosity, 0, 1))

        current_time = Time(time.time(), format='unix')
        if epoch != None:
            self.epoch_time = Time(epoch, format='byear')
        else:
            self.epoch_time = Time(time.time(), format='unix')

        # Verify observing epoch not in past
        assert self.epoch_time >= current_time

        # Guess some parameters which depends on PROGTEMP
        spectrograph_dict = get_progtemp_info(progtemp,
                                              progtemp_dict=self.progtemp_dict)

        obsmode = spectrograph_dict['obsmode']

        assert obsmode in ['MOS']

        assert (spectrograph_dict['red_resolution'] ==
                spectrograph_dict['blue_resolution'])
        resolution = spectrograph_dict['red_resolution']

        red_vph = spectrograph_dict['red_vph']
        blue_vph = spectrograph_dict['blue_vph']

        assert (spectrograph_dict['red_num_exposures'] ==
                spectrograph_dict['blue_num_exposures'])
        num_science_exposures = spectrograph_dict['red_num_exposures']

        assert (spectrograph_dict['red_exp_time'] ==
                spectrograph_dict['blue_exp_time'])
        science_exptime = spectrograph_dict['red_exp_time']

        assert (spectrograph_dict['red_binning_x'] ==
                spectrograph_dict['blue_binning_x'])
        binning_x = spectrograph_dict['red_binning_x']

        # Set the spatial binning from the input of this method
        binning_y = 1

        chained = ('+' in progtemp)
        observation_name = field_name


        if linkedgroup_priority != None:
            logging.warning(f"'linkedgroup_priority' functionality not yet \
                            implemented in OCS. Submission with filled values \
                            may result in WASP Submission Failure")

        if linkedgroup != '':
            linkedgroup_priority = '0'
        else:
            linkedgroup_priority = ''

        if obsgroup == '':
            obsgroup_validity = ''
        else:
            logging.warning(f"'obsgroup' functionality not yet \
                implemented in OCS. Overriding submission values.")

            obsgroup = ''
            obsgroup_validity = ''

        targsrvy_list = list(set([target['TARGSRVY'] for target in self.data]))
        targsrvy_list.sort()

        # Guess some parameters from obstemp

        obsconstraints_dict = get_obstemp_info(obstemp,
                                               obstemp_dict=self.obstemp_dict)

        elevation_min = obsconstraints_dict['elevation_min']
        moondist_min = obsconstraints_dict['moondist_min']
        seeing_max = obsconstraints_dict['seeing_max']
        skybright_max = obsconstraints_dict['skybright_max']
        transparency_min = obsconstraints_dict['transparency_min']

        # Create an OB from the XML template

        ob_xml = MOSOBXML(self.xml_template)

        self._def_max_calibration = int(ob_xml.configure.getAttribute('max_calibration'))
        self._def_max_guide = int(ob_xml.configure.getAttribute('max_guide'))
        self._def_max_sky = int(ob_xml.configure.getAttribute('max_sky'))
        self._def_pa = ob_xml.observation.getAttribute('pa')
        # Remove the elements and attributes which are configure outputs
        ob_xml.remove_configure_outputs()

        # Remove the non-used elements
        ob_xml.remove_non_used_elements()

        # Set the attributes of the root element

        ifu_version = 'WIW{}'.format(get_version())

        mos_path = os.path.dirname(os.path.realpath(__file__))
        mos_version = 'WMW{}'.format(get_version(mos_path))

        comment = '{} {}'.format(ifu_version, mos_version)

        root_attrib_dict = {
            'author': self.header_values['CAT_MAIL'][0],
            'cc_report': ','.join(self.header_values['CAT_CC']),
            'comment': comment,
            'report_verbosity': self.report_verbosity,
            'datamver': self.header_values['DATAMVER'][0]
        }

        ob_xml.set_root_attrib(root_attrib_dict)

        # Set the contents of the spectrograph element
        ob_xml.set_spectrograph(binning_x=binning_x, binning_y=binning_y,
                                resolution=resolution, red_vph=red_vph,
                                blue_vph=blue_vph)

        # Set the contents of the exposures element
        ob_xml.set_exposures(num_science_exposures, science_exptime)

        for exposure in ob_xml.exposures.getElementsByTagName('exposure'):
            if exposure.getAttribute('type') == 'science':
                exposure.setAttribute('speed', 'slow')

        # Set the attributes of the observation element
        observation_attrib_dict = {
            'chained': chained,
            'name': observation_name,
            'obstemp': obstemp,
            'obs_mode': obsmode,
            'progtemp': progtemp,
            'trimester': self.header_values['TRIMESTE'][0],
            'linkedgroup' : linkedgroup,
            'linkedgroup_priority' : linkedgroup_priority,
            'tac_alloc': self.header_values['TACALLOC'][0],
            'tac_id': self.header_values['TACID'][0],
            'obsgroup': obsgroup,
            'obsgroup_validity': obsgroup_validity
        }

        if pa is not None:
            observation_attrib_dict['pa'] = self._def_pa

        ob_xml.set_observation(observation_attrib_dict)

        if max_sky != None:
            logging.warning(f"max_sky value of {max_sky} differs from template default of {self._def_max_sky} - may result in WASP Submission Failure")
        else:
            max_sky = self._def_max_sky

        if max_guide != None:
            logging.warning(f"max_guide value of {max_guide} differs from template default of {self._def_max_guide} -  may result in WASP Submission Failure")
        else:
            max_guide = self._def_max_guide

        if max_calibration != None:
            logging.warning(f"max_calibration value of {max_calibration} differs from template default of {self._def_max_calibration} - may result in WASP Submission Failure")
        else:
            max_calibration = self._def_max_calibration

        assert plate in ['PLATE_A', 'PLATE_B']

        num_sky_fibres = max_sky
        plate_max = {'PLATE_A': 960, 'PLATE_B': 940}

        configure_attrib_dict = {
            'max_calibration': max_calibration,
            'max_guide': max_guide,
            'max_sky': max_sky,
            'num_sky_fibres': num_sky_fibres,
            'plate': plate
        }

        ob_xml._set_epoch(self.epoch_time)

        ob_xml.set_configure(configure_attrib_dict)

        # Set the attributes of the obsconstraints element
        obsconstraints_attrib_dict = {
            'elevation_min': elevation_min,
            'moondist_min': moondist_min,
            'seeing_max': seeing_max,
            'skybright_max': skybright_max,
            'transparency_min': transparency_min
        }

        ob_xml.set_obsconstraints(obsconstraints_attrib_dict)

        dithering_attrib_dict = {
            'apply_dither': 0,
            'dither_version': ""
        }
        ob_xml.set_dithering(dithering_attrib_dict)

        offsets = ob_xml.dom.getElementsByTagName('offsets')[0]
        ob_xml._set_attribs(offsets, {'offset_step_ra': "",
                                      'offset_step_dec': ""})

        # Set the contents of the survey element
        ob_xml.set_surveys(targsrvy_list, plate_max[plate])

        if targsrvy_dict != None:
            survey_nodes = ob_xml.surveys.getElementsByTagName('survey')

            for survey in survey_nodes:
                sname = survey.getAttribute('name')
                try:
                    survey.setAttribute('max_fibres',
                                        str(np.minimum(plate_max[plate],
                                                   targsrvy_dict[f"{sname}_max_fibres"])))
                    survey.setAttribute('priority',
                                        str(np.clip(targsrvy_dict[f"{sname}_priority"], 0.0, 1.0)))
                except:
                    raise
                    #logging.warning(f"TARGSRVY {sname} values not present in dict, leaving as default")

        # Set the contents of the fields element
        ob_xml.set_fields(field_ra, field_dec)

        ob_xml.add_guide_and_calib_stars(resolution,
                                         guide_plot_filename=None,
                                         guide_useful_table_filename=None,
                                         mos_num_guide_stars_request=num_guide_stars_request,
                                         mos_num_central_guide_stars=num_central_guide_stars,
                                         mos_min_guide_cut=min_guide_cut,
                                         mos_max_guide_cut=max_guide_cut,
                                         dyn_range_lim=dyn_range_lim,
                                         calib_plot_filename=None,
                                         calib_useful_table_filename=None,
                                         num_calib_stars_request=num_calib_stars_request,
                                         num_central_calib_stars=num_central_calib_stars,
                                         min_calib_cut=min_calib_cut,
                                         max_calib_cut=max_calib_cut,
                                         calib_mixres=calib_mixres,
                                         cache=cache,
                                         guide_mag_lim=99.,
                                         calib_mag_lim=99.)
        

        if selection_dict == None:
            ob_xml._add_table_as_targets(self.data[self.catmask *
                                                   self.prog_obs_cut])

        else:
            for key in selection_dict.keys():
                mask = ((self.data['TARGPROG'] == key) * self.catmask *
                        self.prog_obs_cut)
                filtered_data = self.data[mask]
                scol, slim, smax, sfixed = selection_dict[key]

                if scol in filtered_data.colnames:
                    if smax:
                        if sfixed:
                            downsample = np.argsort(filtered_data[scol])[:slim]
                        else:
                            downsample = (filtered_data[scol] <= slim)
                    else:
                        if sfixed:
                            downsample = np.argsort(filtered_data[scol])[::-1][:slim]
                        else:
                            downsample = (filtered_data[scol] >= slim)

                elif scol == None:
                    downsample = np.random.choice(len(filtered_data), slim, replace=False)

                else:
                    raise Exception(f"{scol} not a valid column for data filtering")

                ob_xml._add_table_as_targets(filtered_data[downsample])

        #ob_xml.populate_targets_with_fits_data(self.data[self.catmask])

        if pa != None:
            ob_xml.observation.setAttribute('pa', pa)
        else:
            ob_xml.observation.setAttribute('pa', self._def_pa)


        ob_xml.clean_target_list()
        # Write the OB XML to a file
        output_path = self._get_output_path(field_name, output_dir=output_dir,
                                            prefix=prefix, suffix=suffix)

        ob_xml.write_xml(output_path)

        if avoidance_list:
            from skyavoidance import gen_avoidance_list, write_avoidance_list

            avoidance_list = gen_avoidance_list(ob_xml, gaia, tycho,
                                                twomass, lga,
                                                avoidance_maglim,
                                                avoidance_scale)

            path = os.path.splitext(output_path)[0]+'.sky.xml'
            avoidance_xml = write_avoidance_list(avoidance_list, path,
                                                 self.xml_datamver)

        return output_path


    def _get_output_path(self, field_name, output_dir='', prefix=None,
                         suffix=''):

        # Set the prefix of the output file if it has not been provided
        if prefix is None:
            prefix = self.default_prefix

        # Choose the first filename which does not exist
        index = 0
        output_path = ''

        while (index < 1) or os.path.exists(output_path):
            index += 1

            output_basename = '{}{}_{:02d}{}.xml'.format(
                prefix, field_name, index, suffix)

            output_path = os.path.join(output_dir, output_basename)

        return output_path


class MOSTrackingCat:
    """
    Convert the target-level data from a MOS catalogue to a set of XMLs.

    This class provides code to take a MOS catalogue containing a set of MOS
    targets and produce protofield XMLs at the desired field centres.

    Parameters
    ----------
    filename : str
        A FITS file containing a MOS cat.

    """
    def __init__(self, filename, exclude_sky=True):

        # Save the filename
        self.filename = filename

        # Set the default prefix to be used for output XMLs
        self.default_prefix, extension = os.path.splitext(filename)

        if extension == '.xml':
            xmlcat = self.read_xml(filename, exclude_sky)
            self.configured = xmlcat[xmlcat['FIBREID'] > 0]
            self.unconfigured = xmlcat[xmlcat['FIBREID'] <= 0]

            self.configured['NCONFIG'] = np.ones(len(self.configured), dtype='int')

        elif extension == '.fits':
            fitscat = Table.read(filename, format='fits')
        else:
            raise Exception(f"'{extension}' not recognised extension")

    def read_xml(self, inxml, exclude_sky=True, verbose=False):
        """Read WEAVE OB XML into a fits table

        Parameters
        ----------

        inxml : str
            Path to input xml file
        exclude_sky : bool (True)
            Ignore sky fibres when constructing target catalogues
        verbose : bool
            Print additional information during XML parsing
        """

        coltypes_t = OrderedDict([("CNAME", "32A"),
                                  ("TARGDEC", "1D"),
                                  ("TARGRA", "1D"),
                                  ("TARGSRVY", "16A"),
                                  ("TARGPROG", "40A"),
                                  ("TARGCAT", "16A"),
                                  ("TARGID", "26A"),
                                  ("TARGNAME", "26A"),
                                  ("TARGPRIO", "1E"),
                                  ("TARGUSE", "16A"),
                                  ("TARGCLASS", "16A"),
                                  ("TARGRA", "1D"),
                                  ("TARGDEC", "1D"),
                                  ("TARGEPOCH", "1E"),
                                  ("TARGPMRA", "1E"),
                                  ("TARGEPMRA", "1E"),
                                  ("TARGPMDEC", "1E"),
                                  ("TARGEPMDEC", "1E"),
                                  ("MAG_G", "1E"),
                                  ("EMAG_G", "1E"),
                                  ("MAG_R", "1E"),
                                  ("EMAG_R", "1E"),
                                  ("MAG_I", "1E"),
                                  ("EMAG_I", "1E"),
                                  ("MAG_GG", "1E"),
                                  ("EMAG_GG", "1E"),
                                  ("MAG_BP", "1E"),
                                  ("EMAG_BP", "1E"),
                                  ("MAG_RP", "1E"),
                                  ("EMAG_RP", "1E"),
                                  ("FIBREID", "1I"),
                                  ("TARGX", "1D"),
                                  ("TARGY", "1D")])


        coltypes_s = OrderedDict([("name","32A"),
                                  ("max_fibres","1I")])

        try:
            dom = xml.dom.minidom.parse(inxml)
        except xml.parsers.expat.ExpatError:
            print("File {0} would not parse".format(inxml))
            return


        # Start by getting the list of targets

        target_list = dom.getElementsByTagName('target')
        #photometry = dom.getElementsByTagName('photometry')
        if (len(target_list) == 0):
            print("File {0} has no targets".format(inxml))
            return

        # Create a dictionary that will hold all of the targets column data

        data = dict();
        for colname in list(coltypes_t.keys()):
            if "MAG" in colname:
                val = []
                for target in target_list:
                    phot = target.getElementsByTagName('photometry')
                    if len(phot) != 1:
                        val.append(np.nan)
                    else:
                        val.append(phot[0].getAttribute(colname.lower()))

                val = np.array(val, dtype='U32')

            else:
                val = np.array([target.getAttribute(colname.lower()) for target in
                                target_list], dtype='U32')
            if verbose:
                print(val)
            blanks = (val == '')

            if verbose:
                print(blanks.sum())
            if not coltypes_t[colname].endswith('A'):
                if coltypes_t[colname].endswith('I'):
                    val[blanks] = '-1'
                else:
                    val[blanks] = 'nan'
            data.update({colname: val})

        """
        for target in target_list:

            attrs = target.attributes
            n = attrs.length
            for i in range(n):
                name = attrs.item(i).name
                val = attrs.item(i).value
                if (name not in data):
                    data.update({name: []})
                data.get(name).append(val)
        """

        # Create FITS columns for the targets table

        cols = []
        for colname in list(coltypes_t.keys()):
            if verbose:
                print(colname)
                print(data.get(colname))
            c = fits.Column(name=colname,format=coltypes_t.get(colname),
                            array=data.get(colname))
            cols.append(c)

        observation = dom.getElementsByTagName('observation')[0]
        progtemp = observation.attributes['progtemp'].value.split('.')[0]
        #obsconstraints = dom.getElementsByTagName('obsconstraints')[0]
        obstemp = observation.attributes['obstemp'].value

        progtemp_col = fits.Column(name='PROGTEMP',
                                   format='8A',
                                   array=[progtemp]*len(c.array))

        cols.append(progtemp_col)

        obstemp_col = fits.Column(name='OBSTEMP',
                                   format='5A',
                                   array=[obstemp]*len(c.array))

        cols.append(obstemp_col)

        # Create the targets table
        tcols = fits.ColDefs(cols)
        thdu_t = fits.BinTableHDU.from_columns(tcols)
        thdu_t.name = "TARGETS"

        # Get the list of surveys

        survey_list = dom.getElementsByTagName('survey')

        # Create a dictionary that will hold all of the surveys column data

        data = dict();
        for survey in survey_list:
            attrs = survey.attributes
            n = attrs.length
            for i in range(n):
                name = attrs.item(i).name
                val = attrs.item(i).value
                if (name not in data):
                    data.update({name : []})
                data.get(name).append(val)

        # Create FITS columns for the survey table

        cols = []
        for colname in list(coltypes_s.keys()):
            c = fits.Column(name=colname,format=coltypes_s.get(colname),
                            array=data.get(colname))
            cols.append(c)

        # Create the survey table but do nothing

        tcols = fits.ColDefs(cols)
        thdu_s = fits.BinTableHDU.from_columns(tcols)
        thdu_s.name = "surveys"

        xml_table = Table(thdu_t.data)

        if exclude_sky:
            xml_table = xml_table[xml_table['TARGUSE'] != 'S']

        return xml_table

    def add_ob(self, inxml, unique_col='CNAME', exclude_sky=True,
               rtol=0, atol=1e-5):
        """ Add new configured WEAVE OB XML to target tracking tables

        Parameters
        ----------

        inxml : str
            Path to input xml file
        unique_col : str, default='CNAME'
            Target property to use when matching configured targets - used in
            combination with OBSTEMP and PROGTEMP to match targets configured
            in the same observing mode.
        exclude_sky : bool, default=True
            Ignore sky fibres when constructing target catalogues
        verbose : bool
            Print additional information during XML parsing
        """

        ob_fits = self.read_xml(inxml)
        if exclude_sky:
            ob_fits = ob_fits[ob_fits['TARGUSE'] != 'S']

        new_configured = ob_fits[ob_fits['FIBREID'] > 0]
        new_configured['NCONFIG'] = np.ones(len(new_configured), dtype='int')
        new_unconfigured = ob_fits[ob_fits['FIBREID'] <= 0]

        # Generate unique target + observing mode IDs
        unique_xml = np.array([str(row[unique_col]) + '_' + row['PROGTEMP'] +
                               '_' + row['OBSTEMP'] for row in new_configured])
        unique_configured = np.array([str(row[unique_col]) + '_' +
                                      row['PROGTEMP'] + '_' + row['OBSTEMP'] for
                                      row in self.configured])

        ra_configured = np.copy(self.configured['TARGRA'])
        dec_configured = np.copy(self.configured['TARGDEC'])

        for ir, row in enumerate(new_configured):
            if unique_xml[ir] in unique_configured:
                # Find matches based on unique ID
                row_match = (unique_xml[ir] == unique_configured)

                # Verify that coordinates match
                row_match *= np.isclose(ra_configured,
                                        row['TARGRA'],
                                        rtol=rtol, atol=atol)
                row_match *= np.isclose(dec_configured,
                                        row['TARGDEC'],
                                        rtol=rtol, atol=atol)
                rx = np.where(row_match)[0][0]
                self.configured['NCONFIG'][rx] += 1
            else:
                self.configured.add_row(row)

        # Append unconfigured targets - final desired behaviour TBD
        self.unconfigured = vstack([self.unconfigured, new_unconfigured])



def _getimages(ra, dec, size=240, filters="grizy"):
    """Query ps1filenames.py service to get a list of images.

    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """
    # <codecell> Code from PanSTARRS https://ps1images.stsci.edu/ps1image.html
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = ("{service}?ra={ra}&dec={dec}&size={size}&format=fits"
           "&filters={filters}").format(**locals())
    table = Table.read(url, format='ascii')
    return table


def _geturl(ra, dec, size=240, output_size=None, filters="grizy",
           format="jpg", color=False):
    """
    Get URL for images in the table.

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URL for single-filter grayscale images.
    Returns a string with the URL
    """
    if color and format == "fits":
        raise ValueError(
            "color images are available only for jpg or png formats")
    if format not in ("jpg", "png", "fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = _getimages(ra, dec, size=size, filters=filters)
    url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           "ra={ra}&dec={dec}&size={size}&format={format}").format(**locals())
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0, len(table)//2, len(table)-1]]
        for i, param in enumerate(["red", "green", "blue"]):
            url = url + "&{}={}".format(param, table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase+filename)
    return url


def check_sky_photometry_ps(obxml, output_dir,
                            size_pix=40, filters="gri"):

    fibre_data = MOSTrackingCat(obxml, exclude_sky=False)
    sky_fibres = fibre_data.configured[fibre_data.configured['TARGUSE'] == 'S']
    field = os.path.splitext(os.path.split(obxml)[1])[0]
    # if isinstance(obxml, MOSOBXML):
    #     ob = obxml
    #
    # else os.path.isfile(obxml):
    #     try:
    #         ob = MOSOBXML(obxml)
    #     except:
    #         logging.error('File {} would not parse or is not a MOSOBXML instance'.format(self.filename))
    #         raise

    from photutils import CircularAperture, aperture_photometry

    ps1_pixel = 0.25  # arcsec
    fibre_radius = 1.305/2.  # arcsec (1.305 arcsec diameter)
    fibre_area = np.pi*fibre_radius**2

    ra_Gaia = sky_fibres['TARGRA']
    dec_Gaia = sky_fibres['TARGDEC']
    coords_Gaia = SkyCoord(ra=ra_Gaia, dec=dec_Gaia,
                           unit='deg', frame=ICRS())
    coords_PS1 = coords_Gaia.transform_to(FK5(equinox='J2000'))
    ra = coords_PS1.ra.value
    dec = coords_PS1.dec.value

    for band in filters:
        colname = 'MAG_'+band.upper()
        sky_fibres[colname] = np.ones_like(ra)*np.nan

        for i, fibre in enumerate(sky_fibres):
            filename = f"{field}_{band}_{size_pix}_{fibre['FIBREID']}.fits"
            full_path = os.path.join(output_dir, filename)
            if os.path.isfile(full_path):
                logging.info('{} already exists'.format(filename))
                hdu = fits.open(full_path)
            else:
                logging.info('Downloading {}'.format(filename))
                fitsurl = _geturl(ra[i], dec[i], size=size_pix,
                                 filters=band, format="fits")
                for url in fitsurl:
                    band = url[-13]
                    # logging.debug(filename, band)
                    hdu = fits.open(url)
                    hdu.writeto(full_path, overwrite=True)

            zero_point = 25 + 2.5*np.log10(hdu[0].header['exptime'])
            wcs_info = WCS(hdu[0].header)
            img = hdu[0].data

            coords_pix = utils.skycoord_to_pixel(coords_PS1[i], wcs_info,
                                                 origin=0, mode='all')

            radius=fibre_radius/ps1_pixel

            bg_pos = np.random.randint(radius, size_pix-radius,
                                       size=(100, 2))
            ap_pos = np.concatenate([np.array(coords_pix).transpose()[None, :], bg_pos])

            logging.debug('radius={} pixels'.format(radius))

            apertures = CircularAperture(ap_pos,
                                         r=radius)
            phot_table = aperture_photometry(img, apertures)
            # good_photo = np.where(phot_table['aperture_sum'] > 0)

            if phot_table['aperture_sum'][0] > 0:
                photo = zero_point - 2.5*np.log10(
                    phot_table['aperture_sum'][0])
            else:
                bg_std = mad_std(phot_table['aperture_sum'][1:])
                bg_median = np.median(img)
                photo = zero_point - 2.5*np.log10(bg_median + bg_std)

            sky_fibres[colname][i] = photo


    return sky_fibres
