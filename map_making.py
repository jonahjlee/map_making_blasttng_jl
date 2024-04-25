# ============================================================================ #
# map_making.py
#
# James Burgoyne jburgoyne@phas.ubc.ca 
# CCAT Prime 2024
#
# Script to make naive combined maps of BLAST-TNG data.
# ============================================================================ #


import os
import re
import gc
import sys
import time
from datetime import datetime
from typing import NamedTuple
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
from scipy.optimize import minimize


# ============================================================================ #
# -CONFIG-
# ============================================================================ #

roach = 1

maps_to_build = ['DF'] # options: ['A', 'P', 'DF']

# platescale, band, psf
platescale = 5.9075e-6 # deg/um = 21.267 arcsec/mm
band = {1:500, 2:250, 3:350, 4:250, 5:250} # um
psf = {1:0.0150, 2:0.0075, 3:0.0105, 4:0.0075, 5:0.0075} # deg
# arcsec: 54, 27, 38, 27, 27
# f/# = 3.87953; D=2.33 (w/ lyot stop)

# map pixel bin sizes to use (determines final resolution)
# beam is 0.01 degrees or 36 arcsec
pixels_per_beam = 2 # 2 in x and y = 4 pixel sampling of beam
x_bin = y_bin = psf[roach]/pixels_per_beam/platescale # um

# KID to use as the reference for shift table calculations
kid_ref = {1:'0100', 2:'', 3:'0003', 4:'', 5:''}[roach]

# source name for SkyCoord
source_name = 'RCW 92'

# data indices
# scan of object RCW 92
slice_i = {1:37125750, 2:0, 3:37141250, 4:0, 5:0}[roach] # RCW 92
cal_i   = slice_i + 516_000 # cal lamp
cal_f   = slice_i + 519_000

# base data directories
dir_root   = '/media/player1/blast2020fc1/fc1/'   # control computer
dir_conv   = dir_root + "converted/"              # control computer

"""
dir_root   = '/Users/james/Documents/Projects' \
             '/2022/p22a__CCATp/map_making/data/' # James' local
dir_conv   = dir_root                             # James' local
"""

# data directories and files
dir_master = dir_conv + 'master_2020-01-06-06-21-22/'
dir_roach  = dir_conv
dir_targ   = dir_root + f'roach_flight/roach{roach}/targ/'
if roach == 1:
    dir_roach   += 'roach1_2020-01-06-06-22-01/'
    dir_targ    += 'Tue_Jan__7_00_55_50_2020/'
elif roach == 2:
    dir_roach   += 'roach2_2020-01-06-06-22-01/'
    dir_targ    += 'Tue_Jan__7_00_55_50_2020/'
elif roach == 3:
    dir_roach   += 'roach3_2020-01-06-06-21-56/'
    dir_targ    += 'Tue_Jan__7_00_55_51_2020/'
elif roach == 4:
    dir_roach   += 'roach4_2020-01-06-06-22-01/'
    dir_targ    += 'Tue_Jan__7_00_55_50_2020/'
elif roach == 5:
    dir_roach   += 'roach5_2020-01-06-06-22-01/'
    dir_targ    += 'Tue_Jan__7_00_55_51_2020/'

# detector layout file
file_layout = dir_root + f'map_making/detector_layouts/layout_roach{roach}.csv'

# KID rejects list
file_rejects = dir_root + f'map_making/kid_rejects/kid_rejects_roach{roach}.dat'

# common-mode file
file_commonmode = f'common_mode_roach{roach}.dat'

# log file
log_file = 'map_making.log'

# single KID maps output directory
dir_single = 'single_maps/'

# map aligning parameters output dir and file
dir_xform = 'align/'
# file_xform = dir_xform + f'align_roach{roach}.npy'
file_source_coords = dir_xform + f'source_coords_roach{roach}.npy'

# first unused KID channel (2469 total used channels)
kid_max = {1:380, 2:474, 3:667, 4:498, 5:450}[roach]

# TOD peak properties for find_peaks for source search
peak_s = 3   # prominence [multiple of noise]
peak_w = 100 # width [indices] 

# Noise finding highpass parameters
noise_cutoff_freq = 10 # Hz
noise_order       = 3



# ============================================================================ #
# ------------
# ============================================================================ #
# -SETUP FUNCS-
# ============================================================================ #


# ============================================================================ #
# Timer
class Timer:
    def __init__(self):
        self.time_i = time.time()
    
    def deltat(self):
        return time.time() - self.time_i
    

# ============================================================================ #
# mem
def mem(s):
    return f"{s}: memory [bytes]: {tracemalloc.get_traced_memory()}"


# ============================================================================ #
# timestamp
def genTimestamp():
    '''String timestamp of current time (UTC) for use in filenames.
    ISO 8601 compatible format.
    '''

    return datetime.utcnow().strftime("%Y-%m-%d-T%H_%M_%SZ")


# ============================================================================ #
# genLog
def genLog(log_file, log_dir):
    '''Generate a log.
    '''
    
    level = logging.INFO
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log   = logging.getLogger('log')
    log.setLevel(level)
    
    # file handler adds to log file
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    
    return log


# ============================================================================ #
# genDirsForRun
def genDirsForRun(suffix, dirs):
    '''Generate needed directories for this run.
    
    suffix: Added to end of base directory created.
    '''
    
    # base directory
    dir_out = os.path.join(os.getcwd(), f"map_{suffix}")
    os.makedirs(dir_out)
    
    for d in dirs:
        os.makedirs(os.path.join(dir_out, d), exist_ok=True)

    return dir_out



# ============================================================================ #
# -COM FUNCS-
# ============================================================================ #


# ============================================================================ #
# loadCommonData
def loadCommonData(roach, dir_master, dir_roach):
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
# findAllKIDs
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
def loadKidRejects(file_rejects):
    '''
    '''

    # load rejects file
    dat = np.loadtxt(file_rejects, delimiter=' ', dtype=str)

    return dat


# ============================================================================ #
# alignMasterAndRoachTods
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
# -KID FUNCS-
# ============================================================================ #


# ============================================================================ #
# loadKIDData
def loadKIDData(roach, kid):
    '''Preps KID I and Q for on-demand loading.
    '''

    I = np.load(dir_roach + f'i_kid{kid}_roach{roach}.npy', 
                allow_pickle=False, mmap_mode='r')
    Q = np.load(dir_roach + f'q_kid{kid}_roach{roach}.npy', 
                allow_pickle=False, mmap_mode='r')
    
    return I, Q


# ============================================================================ #
# getTargSweepIQ
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


# ============================================================================ #
# createAmp
@nb.njit
def createAmp(I, Q):
    '''Calculate amplitude values.'''
    
    return np.sqrt(I**2 + Q**2)


# ============================================================================ #
# createPhase
@nb.njit
def createPhase(I, Q):
    '''Calculate phase values.'''
    
    return np.arctan2(Q, I)


# ============================================================================ #
# df_IQangle
def df_IQangle(I, Q, If, Qf, Ff, i_f0=None):
    '''Calculate df using IQ Angle Method.
    
    I: (1D array of floats) Timestream S21 real component.
    Q: (1D array of floats) Timestream S21 imaginary component.
    If: (1D array of floats) Target sweep S21 real component.
    Qf: (1D array of floats) Target sweep S21 imaginary component.
    Ff: (1D array of floats) Target sweep S21 frequency axis.
    '''
    
    if i_f0 is None:                        # resonant frequency index
        i_f0 = np.argmin(np.abs(If + 1j*Qf)) 
    
    cI = (If.max() + If.min())/2            # centre of target IQ loop
    cQ = (Qf.max() + Qf.min())/2
    
    # target sweep
    If_c, Qf_c = If - cI, Qf - cQ           # shift center to origin
    θf = np.arctan2(Qf_c, If_c)             # find IQ angles
    
    # observations
    I_c, Q_c = I - cI, Q - cQ               # shift origin
    θ = np.arctan2(Q_c, I_c)                # find IQ angles
    
    # adjust frequencies for delta from f0
    Ff0 = Ff - Ff[i_f0]                     # center Ff on f0
    
    # interpolate
    df = np.interp(θ, θf, Ff0, period=2*np.pi)
    
    return df/Ff[i_f0]


# ============================================================================ #
# genTOD
def genTOD(tod_type, I, Q, If, Qf, Ff):
    '''
    '''

    if   tod_type == 'A' : return createAmp(I, Q)
    elif tod_type == 'P' : return createPhase(I, Q)
    # elif tod_type == 'DF': return createΔfx_grad(I, Q, If, Qf)
    elif tod_type == 'DF': return df_IQangle(I, Q, If, Qf, Ff)

    else: raise Exception(f'Error: Invalid tod_type ({tod_type})')



# ============================================================================ #
# -ANALYSIS FUNCS-
# ============================================================================ #


# ============================================================================ #
# invTodIfNegPeaks
def invTodIfNegPeaks(tod, cal_i, cal_f):
    '''Invert TOD if peaks are negative.
    '''
    
    # only use cal lamp region (works best)
    d = tod[cal_i:cal_f]

    med = np.median(d)
    if (med - d.min()) > (d.max() - med):
        tod *= -1
        
    return tod


# ============================================================================ #
# butterFilter
def butterFilter(data, t, btype, cutoff_freq, order):

    f_sampling = 1 / (t[1] - t[0])
    nyquist = 0.5*f_sampling
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    filtered_data = filtfilt(b, a, data)

    return filtered_data


# ============================================================================ #
# todNoise
def todNoise(tod, t, cutoff_freq, order):
    '''Noise (std) in TOD sample.

    tod: (1D array; floats) Time ordered data, e.g. amp, phase, or df.
    t: (1D array; floats) Time array.
    cutoff_freq: (float) The highpass frequency cutoff.
    order: (int) Butter filter order.
    '''

    filtered_tod = butterFilter(tod, t, 'high', cutoff_freq, order)
    
    std = np.std(filtered_tod)

    return std


# ============================================================================ #
# normTod
def normTod(tod, cal_i, cal_f):
    '''Normalize data as 0+epsilon to cal lamp peak.
    
    tod: (1D array; floats) Time ordered data to normalize.
    cal_i/cal_f: (int) Calibration start/end indices.
    '''

    # scale/shift a_old to a_new, b_old to b_new
    
    a_old = np.median(tod) # median yields better true amplitudes
    b_old = tod[cal_i:cal_f].max() # cal lamp max
    
    a_new = sys.float_info.epsilon # ~0
    b_new = 1
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        tod_norm = a_new + (b_new - a_new)*(tod - a_old)/(b_old - a_old)

    return tod_norm


# ============================================================================ #
# detectPeaks
def detectPeaks(a, std, s=3, w=100):
    '''Detect peaks in data array.
    
    a: (1D array of floats) Data (TOD) to find peaks in.
    std: (float) Standard deviation of noise of a.
    s: (float) Multiplication factor of std (sigma) to set peak prominence to.
    w: (float) Min width of peak.
    '''
    
    peak_indices, info = find_peaks(a, prominence=s*std, width=w)
    
    return peak_indices, info


# ============================================================================ #
# weightedMedian
def weightedMedian(data, weights):
    '''Weighted median of data.

    data: (1D array; floats) The data array.
    weights: (1D arrayl floats) Weight at each data point.
    '''

    # Sort data and corresponding weights
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Calculate cumulative sum of weights
    cumulative_weights = np.cumsum(sorted_weights)

    # Find median index
    median_index = np.searchsorted(cumulative_weights, 0.5 * np.sum(weights))

    # Return median value
    return sorted_data[median_index]


# ============================================================================ #
# sourceCoords
def sourceCoords(xx, yy, zz):
    '''Find source (in image coords); smooth then find max pixel.
    
    xx, yy, zz: (2D arrays of floats) The image data (meshgrid style).
    '''
    
    # gaussian smoothing
    smoothing_kernel = 2 # in pixel coords
    nonan_array = np.where(np.isnan(zz), np.nanmedian(zz), zz)
    smoothed_array = gaussian_filter(nonan_array, sigma=smoothing_kernel)
    
    # identify source in smoothed map by max value (pixel coords)
    max_coords = np.unravel_index(np.argmax(smoothed_array), smoothed_array.shape) 
    
    # convert max point to image coords
    x_im, y_im = xx[max_coords], yy[max_coords]

    # im coords have origin at center
    return x_im, y_im


# ============================================================================ #
# sourceCoordsToPixShift
def sourceCoordsToPixShift(x_im, y_im, xx, yy):
    '''
    '''
    
    # bin sizes
    dx = np.abs(xx[0,1] - xx[0,0])
    dy = np.abs(yy[1,0] - yy[0,0])

    # calculate shift in pix coords
    # shift is negative of position
    # divide by bin size to convert to pix deltas
    Δx = -x_im/dx    
    Δy = -y_im/dy

    return Δx, Δy



# ============================================================================ #
# -MAP FUNCS-
# ============================================================================ #


# ============================================================================ #
# sourceCoordsAzEl
def sourceCoordsAzEl(source_name, lat, lon, alt, time):
    '''Tod of az/el coordinates of the source (telescope frame).

    source_name: (str) The source name for SkyCoord.
    lat, lon, alt, time: (1D arrays) Tods to define telescope location.
    '''

    # Define the coordinates of the source
    source = SkyCoord.from_name(source_name)

    # Define the location of the telescope (tod)
    telescope_location = EarthLocation(
        lat=lat*u.deg, lon=lon*u.deg, height=alt*u.m)

    # Create an astropy Time object of Unix time array
    time = np.nan_to_num(time) # deals with any nan or inf
    time_array = Time(time, format='unix_tai') # timestamp w/ us

    # Altitude-Azimuth frame tod
    altaz_frame = AltAz(obstime=time_array, location=telescope_location)

    # Calculate the coordinates of source in el/az frame (tod)
    source_altaz = source.transform_to(altaz_frame)

    # Extract azimuth and elevation from the resulting AltAz object
    az = source_altaz.az.to(u.deg).value
    el = source_altaz.alt.to(u.deg).value
    
    # return as NamedTuple
    class SourceAzEl(NamedTuple): 
        az: np.ndarray
        el: np.ndarray
    return SourceAzEl(az, el)


# ============================================================================ #
# azElOffsets
def azElOffsets(source_azel, az, el):
    '''Generate az/el offset tods from az/el pointing and source coords.

    source_azel: (SourceAzEl) Source az/el tods.
    az, el: (1D array) Pointing az/el tods.
    '''
    
    # generate offsets
    offset_az = source_azel.az - az
    offset_el = source_azel.el - el
    
    # determine correction to pointing
    # since boresight differs from pointing data
    cor_az = np.mean(offset_az)
    cor_el = np.mean(offset_el)
    offset_az -= cor_az
    offset_el -= cor_el
          
    log.info(f"Boresight correction: az={cor_az}, el={cor_el}")
    
    return offset_az, offset_el


# ============================================================================ #
# offsetsTanProj
def offsetsTanProj(x, y, platescale):
    '''Convert offsets in degrees to image plane in um.

    x, y: (1D array of floats) The offsets arrays in degrees.
    platescale: (float) Platescale in deg/um.
    '''
    
    from astropy.wcs import WCS
    
    w = WCS(naxis=2)
    w.wcs.crpix = [0, 0]   # center of the focal plane is tangent point
    w.wcs.crval = [0., 0.] # source is at center in offsets map
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cdelt = [platescale, platescale]
    
    xp, yp = w.wcs_world2pix(x, y, 0)

    return xp, yp


# ============================================================================ #
# genMapAxesAndBins
def genMapAxesAndBins(x, y, x_bin, y_bin):
    '''Generate the 1D bin/edge arrays and 2D map arrays.

    x, y: (1D array; floats) x,y tods, e.g. az/el.
    x_bin, y_bin: (float) x/y bin size for map.
    '''

    # generate map bin arrays
    x_bins = np.arange(np.min(x), np.max(x), x_bin)
    y_bins = np.arange(np.min(y), np.max(y), y_bin)

    # generate map bin edge arrays
    x_edges = np.arange(np.min(x), np.max(x) + x_bin, x_bin)
    y_edges = np.arange(np.min(y), np.max(y) + y_bin, y_bin)

    # generate meshgrid 2D map bin arrays
    xx, yy = np.meshgrid(x_bins, y_bins)

    return xx, yy, x_bins, y_bins, x_edges, y_edges


# ============================================================================ #
# buildSingleKIDMap
def buildSingleKIDMap(tod, x, y, x_edges, y_edges):
    '''Build a 2D map from time ordered data for a single KID.

    x, y: (1D arrays; floats) The time ordered positional data, e.g. az/el.
    TOD: (1D arrays; floats) TOD values, e.g. A, P, DF.
    '''

    zz0, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    zz, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=tod)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        zz = np.divide(zz, zz0)
    # zz /= zz0

    zz[zz == 0] = np.nan

    return zz.T


# ============================================================================ #
# nansum
def nansum(arr, axis=0):
    '''Custom nansum. 
    Nan acts as zero with other values, otherwise stays as nan if all nan values.
    Neither numpy sum nor nansum behave this way.
    '''
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_arr = np.nanmean(arr, axis=axis)
    nan_mask = np.isnan(mean_arr)
    sum_arr  = np.nansum(arr, axis=axis)
    sum_arr[nan_mask] = np.nan
    
    return sum_arr


# ============================================================================ #
# nanaverage
def nanaverage(arr, weights=None, axis=None):
    # Create a mask where NaN values are True
    mask = np.isnan(arr)
    
    # Replace NaN values with zeros for calculation if weights are provided
    if weights is not None:
        arr = np.where(mask, 0, arr)
    
    # Calculate the sum along the specified axis, ignoring NaN values
    total = np.nansum(arr, axis=axis)
    
    # Count the non-NaN values along the specified axis
    count = np.sum(~mask, axis=axis)
    
    # Handle the case where count is zero to avoid division by zero
    count[count == 0] = 1
    
    # Calculate the average
    if weights is None:
        avg = total / count.astype(float)
    else:
        avg = np.average(arr, weights=weights, axis=axis)
    
    return avg



# ============================================================================ #
# ------------
# ============================================================================ #
# -PRE-LOOP-
# ============================================================================ #

tracemalloc.start()

timer      = Timer()
timestamp  = genTimestamp()
dir_out    = genDirsForRun(timestamp, [dir_xform, dir_single])
log        = genLog(log_file, dir_out)

log.info(f"{roach = }")
log.info(f"{maps_to_build = }")
log.info(f"{x_bin = }")
log.info(f"{y_bin = }")
log.info(f"{kid_ref = }")
log.info(f"{source_name = }")
log.info(f"{slice_i = }")
log.info(f"{cal_i = }")
log.info(f"{cal_f = }")
log.info(f"{dir_root = }")
log.info(f"{dir_master = }")
log.info(f"{dir_roach = }")
log.info(f"{dir_targ = }")
log.info(f"{dir_out = }")
log.info(f"{log_file = }")
log.info(f"{dir_single = }")
log.info(f"{file_source_coords = }")
log.info(f"{file_rejects = }")
log.info(f"{kid_max = }")
log.info(f"{peak_s = }")
log.info(f"{peak_w = }")
log.info(f"{noise_cutoff_freq = }")
log.info(f"{noise_order = }")

print(f"Creating maps: {maps_to_build} in roach {roach}.")
print(f"See log file for more info.")
print(f"Output is in: {dir_out}")

print("Performing pre-KID processing: ", flush=True)

prntPad = "   "

print(prntPad + "Loading common data... ", end="", flush=True)

# load the common data
dat_raw = loadCommonData(roach, dir_master, dir_roach)
dat_targs, Ff = loadTargSweepsData(dir_targ)

print("Done.", flush=True)
print(prntPad + "Aligning timestreams... ", end="", flush=True)

# NOTE: We could load using memmap and slice before processing
# This would reduce memory overhead and speed up the process
# But need to look at align process carefully

# align the master and roach data
dat_aligned, dat_align_indices = alignMasterAndRoachTods(dat_raw)

print("Done.", flush=True)

# slice the common data for this map
# note that cal lamp is removed in these slices
# so master tods will be out of sync with roach tods for a while
dat_sliced = {
    field: dat_aligned[field][slice_i:cal_i].copy() 
    for field in dat_aligned}

# free memory and force collection (these tods are large)
del(dat_raw, dat_aligned)
gc.collect()

print(prntPad + "Determining source coordinates in az/el... ", end="", flush=True)

# source coordinates in az/el telescope frame
source_azel = sourceCoordsAzEl( # source_azel.az, source_azel.el
    source_name, 
    dat_sliced['lat'], dat_sliced['lon'], 
    dat_sliced['alt'], dat_sliced['time'])

print("Done.", flush=True)
print(prntPad + "Generating az/el offset coordinates... ", end="", flush=True)

# generate x and y, the az/el offset tods
x, y = azElOffsets(source_azel, dat_sliced['az'], dat_sliced['el'])

print("Done.", flush=True)
print(prntPad + "Converting offsets to image plane... ", end="", flush=True)

# convert offsets in degrees to um on image plane
x, y = offsetsTanProj(x, y, platescale)

print("Done.", flush=True)
print(prntPad + "Generating map axes... ", end="", flush=True)

# generate map bins and axes
xx, yy, x_bins, y_bins, x_edges, y_edges = genMapAxesAndBins(
    x, y, x_bin, y_bin)

print("Done.", flush=True)
print("Determining KIDs to use... ", end="", flush=True)

# kids to use
kids = findAllKIDs(dir_roach) # all in dir_roach; sorted

print(f"found {len(kids)}.")
log.info(f"Found {len(kids)} KIDs to use.")

# move ref kid so it's processed first
kids.remove(kid_ref)
kids.insert(0, kid_ref)

# load KID rejects
try: kid_rejects = loadKidRejects(file_rejects)
except: kid_rejects = []



# ============================================================================ #
# -c(t) LOOP-
# ============================================================================ #
  
tod_sum = None
tod_sum_cnt = 0
    
print("Calculating common-mode... ", end="", flush=True)
log.info(f"COMMON-MODE CALCULATION -------")
log.info(f"delta_t = {timer.deltat()}")

for kid in kids:
    
    if (int(kid) >= kid_max) or (kid in kid_rejects):
        continue
        
    try:
        # the following is a condensed version of what's done in KID-LOOP
        # look there for comments
        I, Q = loadKIDData(roach, kid)
        targ = getTargSweepIQ(kid, dat_targs)
        I_slice = I[dat_align_indices[slice_i:cal_f]]
        Q_slice = Q[dat_align_indices[slice_i:cal_f]]
        del(I, Q)
        tod = genTOD('DF', I_slice, Q_slice, *targ, Ff) # harcoded DF type
        tod = normTod(tod, cal_i - slice_i, cal_f - slice_i)
        tod = tod[:cal_i - slice_i]
        
        # add this tod to summation tod
        if tod_sum is None:
            tod_sum = tod
        else:
            tod_sum += tod
            
        # track number of tods that contributed for mean
        tod_sum_cnt += 1
        
    except Exception as e:
        log.info(f"KID = {kid}:")
        log.error(f"{e}\n{traceback.format_exc()}")
           
commonmode = tod_sum/tod_sum_cnt

np.save(os.path.join(dir_out, file_commonmode), commonmode)

print(f"Done with {tod_sum_cnt} KIDs. ", flush=True)
log.info(f"COMMON-MODE CALCULATION DONE -------")
log.info(f"delta_t = {timer.deltat()}")

del(tod_sum, tod_sum_cnt)



# ============================================================================ #
# -KID LOOP-
# ============================================================================ #

print("Processing KIDs timestreams:", flush=True)
print("(.: success, r: on reject list, l: loading fail, t: TOD processing fail, m: single KID map production fail, s: source coordinates determination fail)")

source_coords_all = {tod_type: {} for tod_type in maps_to_build}
maps_all = {tod_type: {} for tod_type in maps_to_build}

kids_per_map = {} # kids used per map
kid_cnt = 0       # total kids processed (kid loops)

for kid in kids:
    kid_cnt += 1

    log.info(f"KID = {kid} ---------------------")

    # do not proceed with KID channels above max
    if int(kid) >= kid_max:
        log.info(f"Unused channels past here, skipping.")
        break

    # do not proceed with KID channels in reject list
    if kid in kid_rejects:
        print("r", end="", flush=True)
        log.info(f"This KID is on the reject list, skipping.")
        continue

    log.info(f"delta_t = {timer.deltat()}")
    log.info(f"kid count = {kid_cnt+1}")
    log.info(mem('')) # check mem usage per loop

# ============================================================================ #
#   load KID data
        
    try:

        # load I and Q (memmap)
        I, Q = loadKIDData(roach, kid)

        # load target sweep
        targ = getTargSweepIQ(kid, dat_targs)

        # slice and align (include cal lamp in slice for now)
        I_slice = I[dat_align_indices[slice_i:cal_f]] # slicing align indices
        Q_slice = Q[dat_align_indices[slice_i:cal_f]]

        del(I, Q)

    # this kid is a writeoff, move to next
    except Exception as e:
        log.error(f"{e}\n{traceback.format_exc()}")

        print("l", end="", flush=True)
        continue


# ============================================================================ #
#   -MAP LOOP-
    
    df_success = False # df map processed status

    for tod_type in maps_to_build:

        if kid_cnt-1 == 0: # instantiate all of these
            kids_per_map[tod_type] = 0

        log.info(f"Processing map type {tod_type}")

# ============================================================================ #
#     tod

        try:

            # build tod
            tod = genTOD(tod_type, I_slice, Q_slice, *targ, Ff)
            # genTOD(tod_type, I, Q, If, Qf, Ff) # need Ff

            # invert tod data if peaks are negative
            # tod = invTodIfNegPeaks(tod, cal_i - slice_i, cal_f - slice_i)

            # normalize tod data to calibration lamp
            # make sure tod peaks are positive first
            tod = normTod(tod, cal_i - slice_i, cal_f - slice_i)

            # slice away calibration region
            # brings master and roach tods in sync
            tod = tod[:cal_i - slice_i]
            
            # remove common-mode
            if tod_type == 'DF': # only doing this for DF
                tod -= commonmode

            # determine tod noise
            tod_noise  = todNoise(
                tod, dat_sliced['time'], noise_cutoff_freq, noise_order)
            
        except:
            log.warning(f"TOD ({tod_type}) generation failed.")
            print("t", end="", flush=True)
            continue

# ============================================================================ #
#     single map
        
        try:

            # build the binned pixel maps
            zz  = buildSingleKIDMap(tod, x, y, x_edges, y_edges)

        except:
            log.warning(f"Single KID map ({tod_type}) generation failed.")
            print("m", end="", flush=True)
            continue

# ============================================================================ #
#     source coords
        
        try:
            source_coords = sourceCoords(xx, yy, zz) # x_im, y_im
            source_coords_all[tod_type][kid] = source_coords
            log.info(f"source_coords={source_coords}")

        except:
            log.warning(f"Source coordinate determination failed.")
            print("s", end="", flush=True)
            continue
                
                
# ============================================================================ #
#     map post

        # store this map for combining
        maps_all[tod_type][kid] = zz

        # output this map to file
        out  = np.array([xx, yy, zz])
        fname = f"map_{tod_type}_kid_{kid}"
        np.save(os.path.join(dir_out+'/'+dir_single, fname), out)

        kids_per_map[tod_type] += 1

        if tod_type == 'DF':
            df_success = True

# ============================================================================ #
#   kid post

    if df_success: # df map was generated for this kid
        print(".", end="", flush=True) # success

        # # print stmt every 100 kids processed
        # if kid_cnt % 100 == 0 and kid_cnt > 0:
        #     print(kid_cnt, end="", flush=True)

    else:
        print(f"({kid})", end="", flush=True) # no df map: fail

    gc.collect()
    

    
# ============================================================================ #
# -POST-LOOP-
# ============================================================================ #

print(" Done.")
print(f"Total number of KIDs contributing to maps = {kids_per_map}/{kid_cnt}")

# combine single KID maps (mean)
for tod_type in maps_to_build:
    print(f"Combing {len(maps_all[tod_type])} {tod_type} maps... ", end="")

    # kids used for this map
    kids = sorted(source_coords_all[tod_type].keys())


# ============================================================================ #
# Source Shifts
    
    shifts = {
        kid: sourceCoordsToPixShift(xy[0], xy[1], xx, yy)
        for kid, xy in source_coords_all[tod_type].items()}

    # save shifts to file
    file_shifts = os.path.join(dir_out+'/'+dir_xform, f'shifts_{tod_type}.npy')
    np.save(file_shifts, shifts)


# ============================================================================ #
# Combine Maps
        
    # translate maps ([::-1] to deal with numpy reversed x/y?)
    zz_xformed = [
        shift(maps_all[tod_type][kid], shifts[kid][::-1], cval=np.nan, order=0)
        for kid in kids]

    # combine maps
    # zz_combined = np.nanmean(zz_xformed, axis=0)
    zz_weights = [1 / np.nanstd(a) for a in zz_xformed]
    zz_combined = nanaverage(zz_xformed, axis=0, weights=zz_weights)

    # output to file
    file  = os.path.join(dir_out, f"map_{tod_type}")
    np.save(file, [xx, yy, zz_combined])
    print("Done.")

print(f"Done in {timer.deltat():.0f} seconds.")