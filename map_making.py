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
import numpy as np
import numba as nb
from datetime import datetime
from scipy.ndimage import shift
from scipy.signal import find_peaks



# ============================================================================ #
# CONFIG
# ============================================================================ #

log_file = "map_making.log"

roach = 1

# KID to use as the reference for shift table calculations
kid_ref = {1:'0100', 2:'', 3:'0070', 4:'', 5:''}[roach]

# ra and dec TOD conversion factors
conv_ra  = 5.5879354e-09
conv_dec = 8.38190317e-08

# data indices
# this should be object RCW 92
slice_i = {1:37125750, 2:0, 3:37141250, 4:0, 5:0}[roach] # RCW 92
cal_i   = slice_i + 516_000 # cal lamp
cal_f   = slice_i + 519_000
noise_i = slice_i + 140_000 # noise region
noise_f = slice_i + 200_000

# TOD peak properties for find_peaks
peak_s = 3   # prominence [multiple of noise]
peak_w = 100 # width [indices] 

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
    dir_targ    = dir_root + f'roach_flight/roach1/targ/Mon_Jan__6_06_00_33_2020/'
elif roach == 2:
    dir_roach   = dir_conv + f''
    dir_targ    = dir_root + f''
elif roach == 3:
    dir_roach   = dir_conv + f'roach3_2020-01-06-06-21-56/'
    dir_targ    = dir_root + f'roach_flight/roach3/targ/Mon_Jan__6_06_00_34_2020/'
elif roach == 4:
    dir_roach   = dir_conv + f''
    dir_targ    = dir_root + f''
elif roach == 5:
    dir_roach   = dir_conv + f''
    dir_targ    = dir_root + f''

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
def loadTargSweep(kid, dir_targ):
    '''Targ sweep for chan.
    '''
    
    kid = int(kid)
    
    pattern = re.compile(r'^\d{9}\.dat$') # e.g. '849913000.dat'
    dat_targ = np.array([
        np.fromfile(os.path.join(dir_targ, f), dtype = '<f')
        for f in os.listdir(dir_targ) if pattern.match(f)])
    I = dat_targ[::2, kid]
    Q = dat_targ[1::2, kid]
    
    return I,Q


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
def loadKIDData(dir_targ, roach, kid):
    '''Loads KID specific data into memory.
    '''
    
    I = np.load(dir_roach + f'i_kid{kid}_roach{roach}.npy').byteswap()
    Q = np.load(dir_roach + f'q_kid{kid}_roach{roach}.npy').byteswap()
    targ  = loadTargSweep(kid, dir_targ)
    
    return I, Q, targ


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

    dat_align = {
        'indices'    : indices,
        'master_time': master_time_a,
        "ra"         : ra_a,
        "dec"        : dec_a,
        'roach_time' : roach_time_aligned,
    }

    return dat_align
   
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
    dIfdff0 = dIfdf[len(dIfdf)//2] # dI(f)/df at f0
    dQfdff0 = dQfdf[len(dQfdf)//2] # assume f0 is centre index
    I_n = I - np.mean(I)           # centre values on 0
    Q_n = Q - np.mean(Q)           #
    
    den = dIfdff0**2 + dQfdff0**2  # 
    
    numx = ((I_n*dIfdff0 + Q_n*dQfdff0))
    Δfx = numx/den
    
    numy = ((Q_n*dIfdff0 - I_n*dQfdff0))
    Δfy = numy/den
    
    return Δfx


# ============================================================================ #
# invTodIfNegPeaks
def invTodIfNegPeaks(tod):
    '''Invert TOD if peaks are negative.
    '''
    
    med = np.median(tod)

    if (med - tod.min()) > (tod.max() - med):
        tod *= -1
        
    return tod


# ============================================================================ #
# todNoise
def todNoise(tod, i_i, i_f):
    '''Noise (std) in TOD sample.

    tod: (1D array; floats) Time ordered data, e.g. amp, phase, or df.
    i_i: (int) Sample start index within TOD.
    i_f: (int) Sample end index within TOD.
    '''

    return np.std(tod[i_i:i_f])


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
# sourceCoords
def sourceCoords(ra, dec, tod, noise, s, w):
    '''Source ra and dec from weighted mean of TOD peaks.

    ra, dec: (1D array; floats) Time ordered data of RA/DEC.
    tod: (1D array; floats) Time ordered data to find source in, e.g. DF.
    noise: (float) Noise in tod.
    s: (float) Prominence of peak in multiples of noise.
    w: (int) Width of peak in tod indices.
    '''

    peak_indices, peak_info = detectPeaks(tod, noise, s, w)
    
    if len(peak_indices) < 1:
        return None

    ra_peaks  = ra[peak_indices]
    dec_peaks = dec[peak_indices]
    
    wts = peak_info['prominences']

    ra_source  = np.sum(ra_peaks*wts)/np.sum(wts)
    dec_source = np.sum(dec_peaks*wts)/np.sum(wts)

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
log.info(f"conv_ra    = {conv_ra}")
log.info(f"conv_dec   = {conv_dec}")
log.info(f"slice_i    = {slice_i}")
log.info(f"cal_i      = {cal_i}")
log.info(f"cal_f      = {cal_f}")
log.info(f"noise_i    = {noise_i}")
log.info(f"noise_f    = {noise_f}")
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

print(f"Creating amplitude, phase, and df maps from all KIDs in roach {roach}.")
print(f"See log file for more info.")
print(f"Output is in: {dir_out}")

print("Performing pre-KID processing... ", end="")

# load the common data
dat_raw = loadCommonData(roach, dir_master, dir_roach, conv_ra, conv_dec)

# align the master and roach data
dat_align = alignMasterToRoachTOD(
    dat_raw['master_time'], dat_raw['roach_time'], dat_raw['ra'], dat_raw['dec'])
dat_align_indices = dat_align['indices']

# slice the common data for this map
# note that cal lamp is removed in these slices
# so master tods will be out of sync with A, P, and DF for a while
TIME = dat_align['roach_time'][slice_i:cal_i]
RA   = dat_align['ra'][slice_i:cal_i]
DEC  = dat_align['dec'][slice_i:cal_i]

# generate map bins and axes
rr, dd, ra_bins, dec_bins, ra_edges, dec_edges = genMapAxesAndBins(
    RA, DEC, ra_bin, dec_bin)

# delete unneeded vars to free up memory
del(dat_raw)
del(dat_align)

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

kid_cnt = 0
source_coords_ref = None
zz_A_ref = None
zz_P_ref = None
zz_DF_ref = None
zz_A_multi = None
zz_P_multi = None
zz_DF_multi = None
zz_A_kid_cnt = None
zz_P_kid_cnt = None
zz_DF_kid_cnt = None

# output intermediary products at predetermined number of kids
# [1,2,4,8,16,32,50,100,200,300,...]
outputAtKidCnt = np.concatenate(([1,2,4,8,16,32,50], np.arange(100, len(kids), 100)))

print("KID progress: ", end="", flush=True)

for kid in kids:
    
    log.info(f"delta_t = {timer.deltat()}")
    log.info(f"KID = {kid} ---------------------")
    log.info(f"kid count = {kid_cnt+1}")
    
    try:
            
# ============================================================================ #
#   I and Q
        
        # load KID data
        I, Q, targ = loadKIDData(dir_targ, roach, kid)

        # align the KID data
        I_align, Q_align = alignIQ(I, Q, dat_align_indices)

        # slice the KID data for this map
        # keep cal lamp in, so will be out of sync with master tods
        I_slice = I_align[slice_i:cal_f]
        Q_slice = Q_align[slice_i:cal_f]

        del(I, Q, I_align, Q_align) # done with these

# ============================================================================ #
#   tods

        # determine amplitude, phase, and df tods
        # amplitude data
        A  = createAmp(I_slice, Q_slice)
        P  = createPhase(I_slice, Q_slice)
        DF = createΔfx_grad(I_slice, Q_slice, *targ)

        del(I_slice, Q_slice) # done with these

        # invert tod data if peaks are negative
        A  = invTodIfNegPeaks(A)
        P  = invTodIfNegPeaks(P)
        DF = invTodIfNegPeaks(DF)

        # normalize tod data to calibration lamp
        # make sure tod peaks are positive first
        A  = normTod(A, cal_i - slice_i, cal_f - slice_i)
        P  = normTod(P, cal_i - slice_i, cal_f - slice_i)
        DF = normTod(DF, cal_i - slice_i, cal_f - slice_i)

        # slice away calibration region
        # brings master and roach tods in sync
        A    = A[:cal_i - slice_i]
        P    = P[:cal_i - slice_i]
        DF   = DF[:cal_i - slice_i]

        # determine tod noise in featureless region
        # TODO: modify this to find std in high pass filtered timestream
        A_noise  = todNoise(A, noise_i - slice_i, noise_f - slice_i)
        P_noise  = todNoise(P, noise_i - slice_i, noise_f - slice_i)
        DF_noise = todNoise(DF, noise_i - slice_i, noise_f - slice_i)
        
# ============================================================================ #
#   single maps

        # build the binned pixel maps
        zz_A  = buildSingleKIDMap(A, RA, DEC, ra_edges, dec_edges)
        zz_P  = buildSingleKIDMap(P, RA, DEC, ra_edges, dec_edges)
        zz_DF = buildSingleKIDMap(DF, RA, DEC, ra_edges, dec_edges)

        # Normalizing TOD now so don't need this
        # # normalize the map values
        # # we lose all absolute reference, but needed to combine
        # # we don't need to keep the originals in memory so overwrite
        # zz_A  = normImage(zz_A)
        # zz_P  = normImage(zz_P)
        # zz_DF = normImage(zz_DF)

        # inverting TOD so I think we don't need this
        # # some have inverted values for unknown reasons
        # # so attempt to invert if necessary so source contains max val
        # zz_A  = invertImageIfNeeded(zz_A)
        # zz_P  = invertImageIfNeeded(zz_P)
        # zz_DF = invertImageIfNeeded(zz_DF)

        # use first kid we process as the reference for alignment etc.
        # if kid_ref is None: kid_ref = kid
        if zz_A_ref is None:  zz_A_ref = zz_A
        if zz_P_ref is None:  zz_P_ref = zz_P
        if zz_DF_ref is None: zz_DF_ref = zz_DF
        
        log.info(f"ref KID = {kid_ref}")

# ============================================================================ #
#   shifts

        # TODO: can we align on tods before making single kid maps?
        # determine pixel shift information from DF
        if dat_shifts is None:   
            source_coords = sourceCoords(RA, DEC, DF, DF_noise, peak_s, peak_w)
            log.info(f"source_coords={source_coords}")
            if source_coords is None:
                raise

            if source_coords_ref is None: # this is ref kid
                source_coords_ref = source_coords
                shift_pix = (0,0)

            # calculate pixel shift relative to ref kid
            else:
                shift_pix = findPixShift(
                    source_coords, source_coords_ref, ra_bins, dec_bins)

        # load pixel shift from file
        else:
            shift_pix = dat_shifts[int(kid)]

        # pixel shift tracking
        shifts.append((kid, *shift_pix))

        log.info(f"shift_pix={shift_pix}")

        # shift the image to align
        zz_A  = shift(zz_A, np.flip(shift_pix), cval=np.nan, order=0)
        zz_P  = shift(zz_P, np.flip(shift_pix), cval=np.nan, order=0)
        zz_DF = shift(zz_DF, np.flip(shift_pix), cval=np.nan, order=0)
        
# ============================================================================ #
#   multi maps

        # add to combined maps, or create if needed  
        zz_A_multi  = ternNansum(zz_A_multi, zz_A) 
        zz_P_multi  = ternNansum(zz_P_multi, zz_P)
        zz_DF_multi = ternNansum(zz_DF_multi, zz_DF)

        # track how many kids had values for each pixel, for mean
        zz_A_kid_cnt  = ternSum(zz_A_kid_cnt, ~np.isnan(zz_A)) 
        zz_P_kid_cnt  = ternSum(zz_P_kid_cnt, ~np.isnan(zz_P))
        zz_DF_kid_cnt = ternSum(zz_DF_kid_cnt, ~np.isnan(zz_DF))
        
        del(zz_A, zz_P, zz_DF)
        gc.collect()

        # number of KIDs processed
        kid_cnt += 1
            
        # output final and certain intermediary products
        if kid_cnt in outputAtKidCnt or kid_cnt == len(kids):

            # divide sum maps by count to get mean maps for output
            zz_A_out  = np.divide(zz_A_multi, zz_A_kid_cnt, where=zz_A_kid_cnt.astype(bool))
            zz_P_out  = np.divide(zz_P_multi, zz_P_kid_cnt, where=zz_P_kid_cnt.astype(bool))
            zz_DF_out = np.divide(zz_DF_multi, zz_DF_kid_cnt, where=zz_DF_kid_cnt.astype(bool))
            # zz_A_out  = zz_A_multi/kid_cnt
            # zz_P_out  = zz_P_multi/kid_cnt
            # zz_DF_out = zz_DF_multi/kid_cnt
            
            # hack to add nans back in
            # I can't figure out why this is needed
            # nans are lost in the divide
            zz_A_out[zz_A_kid_cnt==0] = np.nan
            zz_P_out[zz_P_kid_cnt==0] = np.nan
            zz_DF_out[zz_DF_kid_cnt==0] = np.nan

            ## TODO in this next 3 lines is the bug

            # generate output arrays
            A_out  = np.array([rr, dd, zz_A_out])
            P_out  = np.array([rr, dd, zz_P_out])
            DF_out = np.array([rr, dd, zz_DF_out])

            del(zz_A_out, zz_P_out, zz_DF_out)

            # generate filenames
            file_A  = os.path.join(dir_prods, f"map_A_{kid_cnt}_kids")
            file_P  = os.path.join(dir_prods, f"map_P_{kid_cnt}_kids")
            file_DF = os.path.join(dir_prods, f"map_DF_{kid_cnt}_kids")

            # save maps
            np.save(file_A, A_out)
            np.save(file_P, P_out)
            np.save(file_DF, DF_out)
            
        print(".", end="", flush=True)
        if kid_cnt % 100 == 0:
            print(kid_cnt, end="", flush=True)

    # something broke, move to next KID
    except Exception as e:
        log.info(e)

        print("0", end="", flush=True)
        continue
    
print(" Done.")

    
    
# ============================================================================ #
# POST
# ============================================================================ #

print(f"Total number of KIDs contributing to maps = {kid_cnt}")
print("Saving final maps... ", end="")

# output final maps
file_A  = os.path.join(dir_out, f"map_A")
file_P  = os.path.join(dir_out, f"map_P")
file_DF = os.path.join(dir_out, f"map_DF")
np.save(file_A, A_out)
np.save(file_P, P_out)
np.save(file_DF, DF_out)

np.save(os.path.join(dir_out, "shifts"), shifts)

print(f"Done in {timer.deltat():.0f} seconds.")