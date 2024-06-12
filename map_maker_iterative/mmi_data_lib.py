# ============================================================================ #
# mmi_data_lib.py
#
# James Burgoyne jburgoyne@phas.ubc.ca 
# CCAT Prime 2024
#
# Map Maker Iterative data library. 
# ============================================================================ #


import os
import re
import gc
import sys
import time
from datetime import datetime
from typing import NamedTuple
import functools
import logging
import warnings
import traceback
import tracemalloc
import numpy as np
import numba as nb
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from scipy.ndimage import shift, gaussian_filter
from scipy.signal import butter, filtfilt, find_peaks




# ============================================================================ #
# logThis
def logThis(func):
    '''Decorator to mark func as one to be logged when run.
    '''

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.logThis = True  # Add an attribute to mark the function
    return wrapper


# ============================================================================ #
# loadCommonData
@logThis
def loadMasterData(roach, dir_master, dir_roach):
    '''Loads all the common data into memory.
    '''
    
    # load master fields
    # master_fields = ['time', 'time_usec', 'ra', 'dec', 'el', 'az', 'alt', 'lon', 'lat']
    master_fields = ['time', 'time_usec', 'el', 'az', 'alt', 'lon', 'lat']
    dat_raw = {
        field: np.load(dir_master + field + '.npy')
        for field in master_fields}
    
    # load common roach fields
    dat_raw['roach_time'] = np.load(dir_roach + f'ctime_built_roach{roach}.npy', mmap_mode='r')

    # combine time_usec into time
    dat_raw['time'] = dat_raw['time'].astype('float64') + dat_raw['time_usec']*1e-6
    del(dat_raw['time_usec'])
    
    # coord conversions (as per format file)
    # for field in ['ra', 'dec', 'el', 'az']:
    for field in ['el', 'az']:
        dat_raw[field]  = dat_raw[field].astype('float64')*8.38190317e-08
    for field in ['lon', 'lat']:
        dat_raw[field]  = dat_raw[field].astype('float64')*1.676380634e-07
    
    return dat_raw


# ============================================================================ #
# loadTargSweep
@logThis
def loadTargSweepsData(dir_targ):
    '''Loads and combines the target sweep files.
    
    dir_targ: (str or path) The absolute filename str or path.
    '''
    
    # load and combine targ files (If, Qf)
    pattern = r'^\d{9}\.dat$'
    files = os.listdir(dir_targ)
    matched_files = [f for f in files if re.match(pattern, f)]
    sorted_files = sorted(matched_files)
    dat_targs = np.array([
        np.fromfile(os.path.join(dir_targ, f), dtype = '<f')
        for f in sorted_files
    ])
    
    # load frequency file (Ff)
    Ff = np.loadtxt(dir_targ + 'sweep_freqs.dat')
    
    return dat_targs, Ff


# ============================================================================ #
# alignMasterAndRoachTods
@logThis
def alignMasterAndRoachTods(dat_raw):
    '''Interpolate master arrays and align roach.
    '''
    
    # master_fields = ['time', 'ra', 'dec', 'el', 'az', 'alt', 'lon', 'lat']
    master_fields = ['time', 'el', 'az', 'alt', 'lon', 'lat']
    roach_time = dat_raw['roach_time']
    
    # interpolate master fields to roach_time length
    def interp(a, a_match):
        x_old = np.arange(len(a))
        x_new = np.linspace(0, len(a)-1, len(a_match))
        a_interp = np.interp(x_new, x_old, a)
        return a_interp

    dat_aligned = {}
    for field in master_fields:
        dat_aligned[field] = interp(dat_raw[field], roach_time)
        del(dat_raw[field])
        gc.collect()

    # dat_aligned = {
    #     field: interp(dat_raw[field], roach_time)
    #     for field in master_fields}
    
    # indices to align roach tods to master tods
    indices = np.searchsorted(roach_time, dat_aligned['time'], side='left')
    indices[indices == len(roach_time)] = len(roach_time) - 1 # max index bug fix
    
    # use aligned roach time as main time tod
    dat_aligned['time'] = roach_time[indices]
        
    return dat_aligned, indices


# ============================================================================ #
# findAllKIDs
@logThis
def findAllKIDs(directory):
    '''Search given directory for KID files and return set of KID numbers.
    Note that the KID numbers are strings with leading zero, e.g. '0100'.
    '''
    
    files = os.listdir(directory)
    
    # Extract 'kid' values from filenames
    kid_values = set()
    for file in files:
        # Check if the file matches the expected format
        if file.startswith('i_kid') and file.endswith('.npy'):
            # Extract 'kid' value from the filename
            kid = int(file.split('_')[1][4:])
            kid_values.add(kid)
            
    # Sort 'kid' values and format them with leading zeros
    sorted_kid_values = sorted(kid_values)
    sorted_kid_values_strings = [f"{kid:04}" for kid in sorted_kid_values]
    
    return sorted_kid_values_strings


# ============================================================================ #
# loadKidRejects
@logThis
def loadKidRejects(file_rejects):
    '''
    '''

    # load rejects file
    dat = np.loadtxt(file_rejects, delimiter=' ', dtype=str)

    return dat


# ============================================================================ #
# loadKIDData
@logThis
def loadKIDData(roach, kid, dir_roach):
    '''Preps KID I and Q for on-demand loading.
    '''

    I = np.load(dir_roach + f'i_kid{kid}_roach{roach}.npy', 
                allow_pickle=False, mmap_mode='r')
    Q = np.load(dir_roach + f'q_kid{kid}_roach{roach}.npy', 
                allow_pickle=False, mmap_mode='r')
    
    return I, Q


# ============================================================================ #
# getTargSweepIQ
@logThis
def getTargSweepIQ(kid, dat_targs):
    '''Filter roach targs for target sweep for this kid.
    
    kid: (int) The KID number.
    dat_targs: (2D array; floats) Array of all target sweeps for this roach.
    '''
    
    # I = dat_targs[::2, int(kid)]
    # Q = dat_targs[1::2, int(kid)]

    I = dat_targs[:, 2*int(kid)]
    Q = dat_targs[:, 2*int(kid)+1]
    
    return I, Q