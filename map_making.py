# ============================================================================ #
# map_making.py
#
# James Burgoyne jburgoyne@phas.ubc.ca 
# CCAT Prime 2024
#
# Script to make the amplitude, phase, and df maps from all the kids.
# Uses BLAST-TNG data.
# ============================================================================ #


import os
import re
import gc
import sys
import time
import logging
import warnings
import traceback
import numpy as np
import numba as nb
from datetime import datetime
from scipy.ndimage import shift
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import find_peaks

import tracemalloc
tracemalloc.start()
def mem(s):
    return f"{s}: memory [bytes]: {tracemalloc.get_traced_memory()}"



# ============================================================================ #
# CONFIG
# ============================================================================ #

log_file = "map_making.log"

roach = 3

maps_to_build = ['A', 'P', 'DF']

# KID to use as the reference for shift table calculations
kid_ref = {1:'0100', 2:'', 3:'0003', 4:'', 5:''}[roach]

# ra and dec TOD conversion factors
conv_ra  = 5.5879354e-09
conv_dec = 8.38190317e-08

# data indices
# this should be object RCW 92
slice_i = {1:37125750, 2:0, 3:37141250, 4:0, 5:0}[roach] # RCW 92
cal_i   = slice_i + 516_000 # cal lamp
cal_f   = slice_i + 519_000
# noise_i = slice_i + 140_000 # noise region
# noise_f = slice_i + 200_000

# tod lowpass filter parameters
filt_cutoff_freq = 15
filt_order       = 3

# TOD peak properties for find_peaks
peak_s = 3   # prominence [multiple of noise]
peak_w = 100 # width [indices] 

# Noise highpass parameters
noise_cutoff_freq = 10 # Hz
noise_order       = 3

# exclusion tolerances (based on DF tod)
tol_std = 0.1
tol_med = 0.01

# map pixel bin sizes to use (determines final resolution)
ra_bin  = 0.01 # degrees; note RA TOD is converted from hours
dec_bin = 0.01 # degrees
# beam is 0.01 degrees or 36 arcsec
# older maps were 0.015 degrees

# declare directories
dir_root   = '/media/player1/blast2020fc1/fc1/'
dir_conv   = dir_root + 'converted/' # converted npy's
dir_master = dir_conv + 'master_2020-01-06-06-21-22/'
if roach == 1:
    dir_roach   = dir_conv + f'roach1_2020-01-06-06-22-01/'
    dir_targ    = dir_root + f'roach_flight/roach1/targ/Tue_Jan__7_00_55_50_2020/'
elif roach == 2:
    dir_roach   = dir_conv + f'roach2_2020-01-06-06-22-01/'
    dir_targ    = dir_root + f'roach_flight/roach2/targ/Tue_Jan__7_00_55_50_2020/'
elif roach == 3:
    dir_roach   = dir_conv + f'roach3_2020-01-06-06-21-56/'
    dir_targ    = dir_root + f'roach_flight/roach3/targ/Tue_Jan__7_00_55_51_2020/'
elif roach == 4:
    dir_roach   = dir_conv + f'roach4_2020-01-06-06-22-01/'
    dir_targ    = dir_root + f'roach_flight/roach4/targ/Tue_Jan__7_00_55_50_2020/'
elif roach == 5:
    dir_roach   = dir_conv + f'roach5_2020-01-06-06-22-01/'
    dir_targ    = dir_root + f'roach_flight/roach5/targ/Tue_Jan__7_00_55_51_2020/'

# map shifts file
dir_shifts = dir_conv + 'shifts/'
file_shifts = dir_shifts + f'shifts_roach{roach}.npy'



# ============================================================================ #
# FUNCTIONS
# ============================================================================ #


# ============================================================================ #
# timestamp
def genTimestamp():
    '''String timestamp of current time (UTC) for use in filenames.
    ISO 8601 compatible format.
    '''

    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


# ============================================================================ #
# genLog
def genLog(log_file, log_dir):
    '''Generate a logging log.
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
    
    # # stream handler prints to console
    # stream_handler = logging.StreamHandler()
    # stream_handler.setLevel(level)
    # stream_handler.setFormatter(formatter)
    # log.addHandler(stream_handler)
    
    return log


# ============================================================================ #
# Timer
class Timer:
    def __init__(self):
        self.time_i = time.time()
    
    def deltat(self):
        return time.time() - self.time_i

    
# ============================================================================ #
# genDirsForRun
def genDirsForRun(suffix):
    '''Generate needed directories for this run.
    
    suffix: Added to end of base directory created.
    '''
    
    # base directory
    dir_out = os.path.join(os.getcwd(), f"map_{timestamp}")
    os.makedirs(dir_out)
    
    # products directory
    dir_prods = os.path.join(dir_out, "prods")
    os.makedirs(dir_prods)
    
    return dir_out, dir_prods


# ============================================================================ #
# loadCommonData
def loadCommonData(roach, dir_master, dir_roach, conv_ra, conv_dec):
    '''Loads all the common data into memory.
    '''
     
    # load in data
    master_ra        = np.load(dir_master + 'ra.npy')
    master_dec       = np.load(dir_master + 'dec.npy')
    master_time      = np.load(dir_master + 'time.npy')
    master_time_usec = np.load(dir_master + 'time_usec.npy')
    roach_time       = np.load(dir_roach + f'ctime_built_roach{roach}.npy').byteswap()
    # byteswap is needed because conversion didn't account for endianness correctly

    # create a combined time field
    master_time = master_time.astype('float64')
    master_time += master_time_usec*1e-6
    del(master_time_usec)
    # note that 100 indices is 1 sec in these master files
    
    # adjust ra and dec as per format file
    master_ra  = master_ra.astype('float64')
    master_dec = master_dec.astype('float64')
    master_ra  *= conv_ra
    master_dec *= conv_dec

    # adjust ra from hours to degrees
    master_ra *= 15
    
    # data to return
    dat_raw = {
        'master_time': master_time,
        "ra"         : master_ra,
        "dec"        : master_dec,
        'roach_time' : roach_time,
    }
    
    return dat_raw


# ============================================================================ #
# loadTargSweep
def loadTargSweepsData(dir_targ):
    '''Targ sweep for chan.
    
    dir_targ: (str or path) The absolute filename str or path.
    '''
    
    pattern = r'^\d{9}\.dat$'
    files = os.listdir(dir_targ)
    matched_files = [f for f in files if re.match(pattern, f)]
    sorted_files = sorted(matched_files)
    
    dat_targs = np.array([
        np.fromfile(os.path.join(dir_targ, f), dtype = '<f')
        for f in sorted_files
    ])
    
    return dat_targs


# ============================================================================ #
# getTargSweepIQ
def getTargSweepIQ(kid, dat_targs):
    '''Filter roach targs for target sweep for this kid.
    
    kid: (int) The KID number.
    dat_targs: (2D array; floats) Array of all target sweeps for this roach.
    '''
    
    I = dat_targs[::2, int(kid)]
    Q = dat_targs[1::2, int(kid)]
    
    return I, Q


# ============================================================================ #
# loadShiftsData
def loadShiftsData(file_shifts):
    '''Load the pre-determined resonator positional shift data.

    file_shifts: (str) The absolute filename of the shift information file.
    '''

    dat_shifts = np.load(file_shifts, allow_pickle=True) # kid: (shift_x, shift_y)
     
    # convert to dictionary
    dat_shifts = {
        int(dat_shifts[i][0]): (float(dat_shifts[i][1]), float(dat_shifts[i][2]))
        for i in range(len(dat_shifts))
    }

    return dat_shifts


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
# loadKIDData
def loadKIDData(roach, kid):
    '''Loads KID specific data into memory.
    '''
    
    I = np.load(dir_roach + f'i_kid{kid}_roach{roach}.npy').byteswap()
    Q = np.load(dir_roach + f'q_kid{kid}_roach{roach}.npy').byteswap()
    
    return I, Q


# ============================================================================ #
# alignMasterToRoachTOD
@nb.njit
def _alignMasterToRoachTOD(master_time, roach_time, ra, dec):
    '''Interpolate master arrays and align to roach time.
    '''
    
    # interpolate master TOD
    x = np.arange(len(master_time))
    new_x = np.linspace(x[0], x[-1], len(roach_time))
    master_time_a = np.interp(new_x, x, master_time)#, left=np.nan, right=np.nan)
    ra_a   = np.interp(new_x, x, ra)#, left=np.nan, right=np.nan)
    dec_a  = np.interp(new_x, x, dec)#, left=np.nan, right=np.nan)
    
    # assumes both arrays are sorted ascending...
    indices = np.searchsorted(roach_time, master_time_a, side='left')
    # indices = np.digitize(master_time_a, dat['roach_time'], right=True)
    
    # fixed max index bug from np.searchsorted
    indices[indices == len(roach_time)] = len(roach_time) - 1 
    
    roach_time_aligned = roach_time[indices]
     
    return indices, master_time_a, ra_a, dec_a, roach_time_aligned

def alignMasterToRoachTOD(master_time, roach_time, ra, dec):
    ret = _alignMasterToRoachTOD(master_time, roach_time, ra, dec)

    indices, master_time_a, ra_a, dec_a, roach_time_aligned = ret

    return indices, ra_a, dec_a, roach_time_aligned
   
# ============================================================================ #
# alignIQ
@nb.njit
def alignIQ(I, Q, indices):
    '''Align I and Q to master.
    '''
    
    I_align = I[indices]
    Q_align = Q[indices]
    
    return I_align, Q_align


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
# createΔfx_grad
def createΔfx_grad(I, Q, If, Qf):
    '''Calculate Δfx from 'gradient' method.'''
        
    dIfdf = np.diff(If)/1e3        # Δx is const. so Δy=dI/df
    dQfdf = np.diff(Qf)/1e3        # /1e3 for units
    
    Zf = np.abs(If + 1j*Qf)
    i_f0 = np.argmin(Zf)
    
    dIfdff0 = dIfdf[i_f0]          # dI(f)/df at f0
    dQfdff0 = dQfdf[i_f0]
    
    I_n = I - np.mean(I)           # centre values on 0
    Q_n = Q - np.mean(Q)           #
    
    den = dIfdff0**2 + dQfdff0**2  # 
    
    numx = ((I_n*dIfdff0 + Q_n*dQfdff0))
    Δfx = numx/den
    
    # numy = ((Q_n*dIfdff0 - I_n*dQfdff0))
    # Δfy = numy/den
    
    return Δfx


# ============================================================================ #
# genTOD
def genTOD(tod_type, I, Q, If, Qf):
    '''
    '''

    if tod_type == 'A':
        return createAmp(I, Q)
    
    elif tod_type == 'P':
        return createPhase(I, Q)
    
    elif tod_type == 'DF':
        return createΔfx_grad(I, Q, If, Qf)
    
    raise Exception(f'Error: Invalid tod_type ({tod_type})')


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
def sourceCoords(ra, dec, tod, noise, s, w, t=0.5):
    '''Determine source RA and DEC coordinates.

    ra, dec: (1D array; floats) Time ordered data of RA/DEC.
    tod: (1D array; floats) Time ordered data to find source in, e.g. DF.
    noise: (float) Noise in tod.
    s: (float) Prominence of peak in multiples of noise.
    w: (int) Width of peak in tod indices.
    t: (float) Threshold to filter peaks [tod units]
    '''

    i_peaks, peak_info = detectPeaks(tod, noise, s, w)
    
    if len(i_peaks) < 1:
        return None

    # weighted mean of all peaks
    # wts = peak_info['prominences']
    # ra_peaks  = ra[i_peaks]
    # dec_peaks = dec[i_peaks]
    # ra_source  = np.sum(ra_peaks*wts)/np.sum(wts)
    # dec_source = np.sum(dec_peaks*wts)/np.sum(wts)

    # only most prominent peaks; assumes normalized 0-1
    i_peaks_f = tod[i_peaks] >= t # threshold

    # weighted median of most prominent peaks
    wts = tod[i_peaks][i_peaks_f]
    ra_source = weightedMedian(ra[i_peaks][i_peaks_f], weights=wts)
    dec_source = weightedMedian(dec[i_peaks][i_peaks_f], weights=wts)

    return ra_source, dec_source


# ============================================================================ #
# findPixShift
def findPixShift(p, ref, ra_bins, dec_bins):
    '''Find the bin shift to move point to reference.

    p: (2tuple; floats) Point (RA, DEC).
    ref: (2tuple; floats) Reference point (RA, DEC).
    ra_bins: (1D array; floats) RA map bins.
    dec_bins: (1D array; floats) DEC map bins.
    '''

    def findBin(v, bins):
        return np.argmin(np.abs(v - bins))
    
    bin_ref = (findBin(ref[0], ra_bins), findBin(ref[1], dec_bins))
    bin_p   = (findBin(p[0], ra_bins), findBin(p[1], dec_bins))
    
    return np.array(bin_ref) - np.array(bin_p)


# ============================================================================ #
# genMapAxesAndBins
def genMapAxesAndBins(ra, dec, ra_bin, dec_bin):
    '''Generate the 1D bin arrays and 2D map arrays.

    ra/dec: (1D array; floats) RA/DEC time ordered data [degrees].
    ra_bin/dec_bin: (float) RA/DEC bin size [degrees].
    '''

    # num_ra_bins  = int(np.ceil((ra.max() - ra.min()) / ra_bin))
    # num_dec_bins = int(np.ceil((dec.max() - dec.min()) / dec_bin))

    # ra_edges  = np.linspace(ra.min(), ra.max(), num_ra_bins + 1)
    # dec_edges = np.linspace(dec.min(), dec.max(), num_dec_bins + 1)

    # rr, dd = np.meshgrid(ra_edges, dec_edges)

    ra_bins  = np.arange(np.min(ra), np.max(ra), ra_bin)
    dec_bins = np.arange(np.min(dec), np.max(dec), dec_bin)

    ra_edges  = np.arange(np.min(ra), np.max(ra) + ra_bin, ra_bin)
    dec_edges = np.arange(np.min(dec), np.max(dec) + dec_bin, dec_bin)

    rr, dd = np.meshgrid(ra_bins, dec_bins)

    return rr, dd, ra_bins, dec_bins, ra_edges, dec_edges


# ============================================================================ #
# buildSingleKIDMap
def buildSingleKIDMap(tod, ra, dec, ra_edges, dec_edges):
    '''Build a 2D map from time ordered data for a single KID.

    ra, dec: (1D arrays; floats) The time ordered ra/dec data.
    TOD: (1D arrays; floats) TOD values, e.g. A, P, DF.
    '''

    zz0, _, _ = np.histogram2d(ra, dec, bins=[ra_edges, dec_edges])
    zz, _, _ = np.histogram2d(ra, dec, bins=[ra_edges, dec_edges], weights=tod)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        zz = np.divide(zz, zz0)
    # zz /= zz0

    zz[zz == 0] = np.nan

    return zz.T


# ============================================================================ #
# normImage
@nb.njit
def normImage(image):
    '''Normalize to 0-1.
    '''
    
    min_val = np.nanmin(image)
    max_val = np.nanmax(image)

    norm_image = (image - min_val) / (max_val - min_val)
    
    return norm_image


# ============================================================================ #
# invertImageIfNeeded
@nb.njit
def invertImageIfNeeded(image):
    '''Invert pixel values. Assumes already normalized.
    '''
    
    min_val = np.nanmin(image)
    med_val = np.nanmedian(image)
    max_val = np.nanmax(image)
    
    # if most prominent feature is negative, invert
    invert = True if (max_val - med_val) < (med_val - min_val) else False
    
    if invert:
        image = -1*image + 1
        
    return image


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
# ternNansum
def ternNansum(var, val):
    '''Ternary nansum allowing for var to be None (first loop).
    '''
    
    return val if var is None else nansum([var, val], axis=0)  


# ============================================================================ #
# ternSum
def ternSum(var, val):
    '''Ternary np.sum allowing for var to be None (first loop).
    '''
    
    return val if var is None else np.sum([var, val], axis=0)



# ============================================================================ #
# PRE
# ============================================================================ #

timer              = Timer()
timestamp          = genTimestamp()
dir_out, dir_prods = genDirsForRun(timestamp)
log                = genLog(log_file, dir_out)

log.info(f"log_file   = {log_file}")
log.info(f"roach      = {roach}")
log.info(f"kid_ref    = {kid_ref}")
log.info(f"{maps_to_build = }")
log.info(f"conv_ra    = {conv_ra}")
log.info(f"conv_dec   = {conv_dec}")
log.info(f"slice_i    = {slice_i}")
log.info(f"cal_i      = {cal_i}")
log.info(f"cal_f      = {cal_f}")
log.info(f"{filt_cutoff_freq = }")
log.info(f"{filt_order = }")
log.info(f"{noise_cutoff_freq = }")
log.info(f"{noise_order = }")
log.info(f"{tol_std = }")
log.info(f"{tol_med = }")
log.info(f"ra_bin     = {ra_bin}")
log.info(f"dec_bin    = {dec_bin}")
log.info(f"peak_s     = {peak_s}")
log.info(f"peak_w     = {peak_w}")
log.info(f"dir_out    = {dir_out}")
log.info(f"dir_root   = {dir_root}")
log.info(f"dir_master = {dir_master}")
log.info(f"dir_roach  = {dir_roach}")
log.info(f"dir_targ   = {dir_targ}")
log.info(f"file_shifts = {file_shifts}")

print(f"Creating maps: {maps_to_build} in roach {roach}.")
print(f"See log file for more info.")
print(f"Output is in: {dir_out}")

print("Performing pre-KID processing... ", end="", flush=True)

# load the common data
dat_raw = loadCommonData(roach, dir_master, dir_roach, conv_ra, conv_dec)
dat_targs = loadTargSweepsData(dir_targ)

# align the master and roach data
dat_align_indices, ra_a, dec_a, roach_time_aligned = alignMasterToRoachTOD(
    dat_raw['master_time'], dat_raw['roach_time'], dat_raw['ra'], dat_raw['dec'])

# slice the common data for this map
# note that cal lamp is removed in these slices
# so master tods will be out of sync with A, P, and DF for a while
TIME = roach_time_aligned[slice_i:cal_i].copy()
RA   = ra_a[slice_i:cal_i].copy()
DEC  = dec_a[slice_i:cal_i].copy()

# free memory and force collection
del(dat_raw, ra_a, dec_a, roach_time_aligned)
gc.collect() 

# generate map bins and axes
rr, dd, ra_bins, dec_bins, ra_edges, dec_edges = genMapAxesAndBins(
    RA, DEC, ra_bin, dec_bin)

print("Done.")
print("Determining KIDs to use... ", end="", flush=True)

# pull in all the possible kids from the files in dir_roach
kids = findAllKIDs(dir_roach)

print(f"found {len(kids)}... Done.")
log.info(f"Found {len(kids)} KIDs to use.")

# move ref kid so it's processed first
kids.remove(kid_ref)
kids.insert(0, kid_ref)

# load pre-determined shifts file
try: dat_shifts = loadShiftsData(file_shifts)
except: dat_shifts = None

# track pixel shifts
shifts = [] # (kid, ra, dec)



# ============================================================================ #
# KID LOOP
# ============================================================================ #

# output intermediary products at predetermined number of kids
# [1,2,4,8,16,32,50,100,200,300,...]
outputAtKidCnt = np.concatenate(([1,2,4,8,16,32,50], np.arange(100, len(kids), 100)))

print("KID progress: ", end="", flush=True)

kid_cnts = {}
kid_cnt = 0
source_coords_ref = {}
zz_ref = {}
outs = {}

zz_kid_cnt = {}
zz_multi = {}
for tod_type in maps_to_build:
    zz_kid_cnt[tod_type] = None
    zz_multi[tod_type] = None

for kid in kids:
    
    log.info(f"delta_t = {timer.deltat()}")
    log.info(f"KID = {kid} ---------------------")
    log.info(f"kid count = {kid_cnt+1}")

    log.info(mem('')) # check mem usage per loop
    # TODO increase of ~270 kB per loop
    # not sure if I expect an increase, but this is manageable
    # i.e. 1000 KIDs x 270 kB = 270 MB
    
    try:

# ============================================================================ #
#   target sweep

        targ = getTargSweepIQ(kid, dat_targs)

# ============================================================================ #
#   I and Q

        # load KID data
        I, Q = loadKIDData(roach, kid)

        # stop if I and Q empty
        if np.median(np.abs(I + 1j*Q)) < 100:
            raise Exception(f"No data in I and Q.")

        # align the KID data
        I_align, Q_align = alignIQ(I, Q, dat_align_indices)

        # slice the KID data for this map
        # keep cal lamp in, so will be out of sync with master tods
        I_slice = I_align[slice_i:cal_f]
        Q_slice = Q_align[slice_i:cal_f]
        
        del(I, Q, I_align, Q_align) # done with these


    # this kid is a writeoff, move to next
    except Exception as e:
        log.error(f"{e}\n{traceback.format_exc()}")

        print("o", end="", flush=True)
        continue


# ============================================================================ #
#   MAP LOOP
    
    map_cnt = 0
    df_success = False
    for tod_type in maps_to_build:
        log.info(f"Processing map type {tod_type}")

        # each map type has a separate kid count
        if kid == kid_ref:
            kid_cnts[tod_type] = 0

        try:

# ============================================================================ #
#     tod

            # build tod
            tod = genTOD(tod_type, I_slice, Q_slice, *targ)

            # lowpass filter to remove some noise
            tod = butterFilter(tod, TIME, 'low', filt_cutoff_freq, filt_order)

            # invert tod data if peaks are negative
            tod = invTodIfNegPeaks(tod, cal_i - slice_i, cal_f - slice_i)

            # normalize tod data to calibration lamp
            # make sure tod peaks are positive first
            tod = normTod(tod, cal_i - slice_i, cal_f - slice_i)

            # slice away calibration region
            # brings master and roach tods in sync
            tod = tod[:cal_i - slice_i]

            # determine tod noise in featureless region
            tod_noise  = todNoise(tod, TIME, noise_cutoff_freq, noise_order)

            # exclude KIDs with too much noise after normalization
            if (np.std(tod) > tol_std) or (np.median(tod) > tol_med):
                log.info(f"Over tolerance: {np.std(tod)=}, {np.median(tod)=}")
                continue
                # raise Exception(f"Bad {np.std(tod)=}, {np.median(tod)=}")

# ============================================================================ #
#     single map

            # build the binned pixel maps
            zz  = buildSingleKIDMap(tod, RA, DEC, ra_edges, dec_edges)

            # use first kid we process as the reference for alignment etc.
            if kid == kid_ref:
                zz_ref[tod_type] = zz

# ============================================================================ #
#     shift

            # load pixel shift info from file
            if dat_shifts:
                shift_pix = dat_shifts[int(kid)]

            # or determine pixel shift information from tod
            else:
                source_coords = sourceCoords(
                    RA, DEC, tod, tod_noise, peak_s, peak_w)
                log.info(f"source_coords={source_coords}")
                if source_coords is None:
                    raise Exception(f"Could not determine source coords.")

                if kid is kid_ref:
                    source_coords_ref[tod_type] = source_coords
                    shift_pix = (0,0)

                # calculate pixel shift relative to ref kid
                else:
                    shift_pix = findPixShift(
                        source_coords, source_coords_ref[tod_type], 
                        ra_bins, dec_bins)

            # pixel shift tracking
            shifts.append((kid, *shift_pix))

            log.info(f"shift_pix={shift_pix}")

            # shift the image to align
            zz  = shift(zz, np.flip(shift_pix), cval=np.nan, order=0)


            # output each single map
            # out  = np.array([rr, dd, zz])
            # file  = os.path.join(dir_prods, f"map_{tod_type}_{kid_cnt[tod_type]}_kids")
            # np.save(file, out)
            # kid_cnt[tod_type] += 1

# ============================================================================ #
#     multi map

            # add to combined maps, or create if needed  
            zz_multi[tod_type] = ternNansum(zz_multi[tod_type], zz) 

            # track how many kids had values for each pixel, for mean
            zz_kid_cnt[tod_type] = ternSum(zz_kid_cnt[tod_type], ~np.isnan(zz)) 
        
            del(zz)
            
            # divide sum map by count to get mean map for output
            zz_out  = np.divide(
                zz_multi[tod_type], zz_kid_cnt[tod_type], 
                where=zz_kid_cnt[tod_type].astype(bool))
        
            # hack to add nans back in
            # I can't figure out why this is needed
            # nans are lost in the divide
            zz_out[zz_kid_cnt[tod_type]==0] = np.nan

            # generate output array
            out  = np.array([rr, dd, zz_out])
            outs[tod_type] = out

# ============================================================================ #
#     output

            # output intermediary products
            if (kid_cnts[tod_type]+1 in outputAtKidCnt):
                fname = f"map_{tod_type}_{kid_cnts[tod_type]+1}_kids"
                np.save(os.path.join(dir_prods, fname), out)

            del(zz_out, out)

# ============================================================================ #
#     map wrapup

            if tod_type == 'DF':
                df_success = True

            kid_cnts[tod_type] += 1

        # this map failed, move to next map type
        except Exception as e:
            log.error(f"{e}\n{traceback.format_exc()}")

            continue

# ============================================================================ #
#   kid wrapup

    # done with maps for this kid
    if df_success:
        kid_cnt += 1
        print(".", end="", flush=True)
    else:
        print("o", end="", flush=True)

    if kid_cnt % 100 == 0 and kid_cnt > 0:
        print(kid_cnt, end="", flush=True)

    gc.collect()
    

    
# ============================================================================ #
# POST
# ============================================================================ #

print(" Done.")
print(f"Total number of KIDs contributing to maps = {kid_cnts}")
print("Saving final maps... ", end="")

for tod_type in maps_to_build:

    # output final maps
    file  = os.path.join(dir_out, f"map_{tod_type}")
    np.save(file, outs[tod_type])

np.save(os.path.join(dir_out, "shifts"), shifts)

print(f"Done in {timer.deltat():.0f} seconds.")