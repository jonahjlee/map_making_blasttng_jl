# ============================================================================ #
# map_making.py
#
# James Burgoyne jburgoyne@phas.ubc.ca 
# CCAT Prime 2024
#
# Script to make the amplitude, phase, and df maps from all the kids.
# Uses BLAST-TNG data.
# ============================================================================ #


from datetime import datetime
import os
import time
import logging
import numpy as np
import re
import sys
from matplotlib import pyplot as plt
from scipy.ndimage import shift
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import numba as nb
import cProfile
import warnings



# ============================================================================ #
# CONFIG
# ============================================================================ #

log_file = "map_making.log"

roach = 1

# kid_ref = '0070' # roach 3
kid_ref = '0100' # roach 1

# ra and dec conversion factors
conv_ra  = 5.5879354e-09
conv_dec = 8.38190317e-08

# initial and final data indices to use for this map
# this should be object RCW 92
# slice_i = 37141250 # roach 3
# slice_f = 37734400 # roach 3
slice_i = 37125750 # roach 1
slice_f = 37718610 # roach 1

# map bin sizes to use (determines final resolution)
ra_bin  = 0.001
dec_bin = 0.015

# declare directories
dir_root   = '/media/player1/blast2020fc1/fc1/converted'
dir_master = dir_root + '/master_2020-01-06-06-21-22/'

# roach 3
# dir_roach  = dir_root + f'/roach{roach}_2020-01-06-06-21-56/'
# dir_targ   = f'/media/player1/blast2020fc1/fc1/roach_flight/roach{roach}/targ/Mon_Jan__6_06_00_34_2020/'

# roach 1
dir_roach  = dir_root + f'/roach{roach}_2020-01-06-06-22-01/'
dir_targ   = f'/media/player1/blast2020fc1/fc1/roach_flight/roach{roach}/targ/Mon_Jan__6_06_00_33_2020/'



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
    
    # data to return
    dat_raw = {
        'master_time': master_time,
        'ra'         : master_ra,
        'dec'        : master_dec,
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
        'ra'         : ra_a,
        'dec'        : dec_a,
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
# sliceCommon
def sliceCommon(roach_time, ra, dec, i, f): 
    '''Slice common data to desired region.
    '''
    
    dat_slice = {
        'time'   : roach_time[i:f],
        'ra'     : ra[i:f],
        'dec'    : dec[i:f],
    }

    return dat_slice


# ============================================================================ #
# sliceIQ
def sliceIQ(I, Q, i, f):
    '''Slice KID data to desired region.
    '''
    
    I_slice = I[i:f]
    Q_slice = Q[i:f]
    
    return I_slice, Q_slice


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
# precomputeMeshgrid
def precomputeMeshgrid(ra, dec, ra_bin, dec_bin):
    '''
    '''
    
    r = np.arange(np.min(ra), np.max(ra), ra_bin)
    d = np.arange(np.min(dec), np.max(dec), dec_bin)
    rr, dd = np.meshgrid(r, d)

    return rr, dd, r, d


# ============================================================================ #
# buildSingleKIDMaps
def buildSingleKIDMaps(A, P, DF, ra_indices, dec_indices, zz_nan):
    '''Build the 3 (amplitude, phase, df) maps for a KID.

    A, P, DF: (1D arrays; floats) TOD values.
    ra_indices, dec_indices: (1D arrays; ints) RA and DEC zz indices.
    zz_nan: (2D array; floats) Template output map full of NaNs.
    '''

    zz_A = zz_nan.copy()
    zz_P = zz_nan.copy()
    zz_DF = zz_nan.copy()

    # I can't find a way to vectorize this
    for i in range(zz_nan.shape[0]):
        for j in range(zz_nan.shape[1]):
            det_inds = (ra_indices == j) & (dec_indices == i)
            if np.any(det_inds):
                zz_A[i, j] = np.nanmean(A[det_inds])
                zz_P[i, j] = np.nanmean(P[det_inds])
                zz_DF[i, j] = np.nanmean(DF[det_inds])

    return zz_A, zz_P, zz_DF


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
# brightestPixel
def brightestPixel(image):
    '''Find the indices of the maximum value in the 2D array.
    '''
    
    indices = np.unravel_index(np.nanargmax(image, axis=None), image.shape)
    
    return np.array(indices)


# ============================================================================ #
# gaussian2d
@nb.njit
def gaussian2d(xy, amplitude, yo, xo, sigma_x, sigma_y, theta, offset):
    '''2D gaussian function.
    '''

    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude * np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()


# ============================================================================ #
# fitGaussian
def fit2DGaussian(image_in):
    '''Best 2D gaussian fit to image data.
    '''

    # can't fit on nans
    image = np.nan_to_num(image_in, copy=True, nan=0.0)

    # Get image dimensions
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    
    # Initial guess for the parameters
    #initial_guess = (np.max(image), np.argmax(image) % image.shape[1], np.argmax(image) // image.shape[1], 2., 2., 0.0, np.min(image))
    initial_guess = (np.max(image), *np.unravel_index(image.argmax(), image.shape), 
                     2., 2., 0.0, np.min(image))

    # Fit the 2D Gaussian
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        popt, pcov = curve_fit(gaussian2d, (x, y), image.ravel(), p0=initial_guess)

    # Extract the center coordinates from the fitted parameters
    center_x, center_y = popt[1], popt[2]

    return np.array([center_x, center_y])


# ============================================================================ #
# imageShiftToAlign
def imageShiftToAlign(ref_image, image, fit_func):
    '''Shift needed to align image to reference.
    '''

    return fit_func(ref_image) - fit_func(image)


# ============================================================================ #
# distance
@nb.njit
def distance(point1, point2):
    '''Calculate the Euclidean distance between two points.
    '''
    
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


# ============================================================================ #
# meanPoint
# @nb.njit
def meanPoint(points, range_threshold=2):
    '''Find the mean position of points within a certain range of each other.
    '''
    
    for i, point in enumerate(points):
        
        # Check distance between this point and other points
        nearby_points = [
            p 
            for j, p in enumerate(points) 
            if i != j and distance(point, p) <= range_threshold]
        
        if nearby_points:
            # Calculate mean position of nearby points
            mean_x = (point[0] + sum(p[0] for p in nearby_points)) / (len(nearby_points) + 1)
            mean_y = (point[1] + sum(p[1] for p in nearby_points)) / (len(nearby_points) + 1)
            return mean_x, mean_y

    # No points found within range
    return None


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
# PRE-LOOP
# ============================================================================ #

timer              = Timer()
timestamp          = genTimestamp()
dir_out, dir_prods = genDirsForRun(timestamp)
log                = genLog(log_file, dir_out)

log.info(f"log_file = {log_file}")
log.info(f"roach = {roach}")
log.info(f"conv_ra = {conv_ra}")
log.info(f"conv_dec = {conv_dec}")
log.info(f"slice_i = {slice_i}")
log.info(f"slice_f = {slice_f}")
log.info(f"ra_bin = {ra_bin}")
log.info(f"dec_bin = {dec_bin}")
log.info(f"dir_out = {dir_out}")
log.info(f"dir_root = {dir_root}")
log.info(f"dir_master = {dir_master}")
log.info(f"dir_roach = {dir_roach}")
log.info(f"dir_targ = {dir_targ}")

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
dat_slice = sliceCommon(
    dat_align['roach_time'], dat_align['ra'], dat_align['dec'], slice_i, slice_f)

# delete unneeded vars to free up memory
del(dat_raw)
del(dat_align)

# build the meshgrid axis arrays for the map
rr, dd, rr_bins, dd_bins = precomputeMeshgrid(dat_slice['ra'], dat_slice['dec'], ra_bin, dec_bin)

# build the pixel map bins
# ra_bins = np.arange(rr.min(), rr.max() + ra_bin, ra_bin)
# dec_bins = np.arange(dd.min(), dd.max() + dec_bin, dec_bin)

# build the map bin indices arrays for ra and dec
ra_indices = np.digitize(dat_slice['ra'], rr_bins) - 1
dec_indices = np.digitize(dat_slice['dec'], dd_bins) - 1

# build a final map shaped nan array for use later
zz_nan = np.full_like(dd, np.nan)

print("Done.")
print("Determining KIDs to use... ", end="", flush=True)

# pull in all the possible kids from the files in dir_roach
kids = findAllKIDs(dir_roach)

print(f"found {len(kids)}... Done.")
log.info(f"Found {len(kids)} KIDs to use.")

# move ref kid so it's processed first
kids.remove(kid_ref)
kids.insert(0, kid_ref)



# ============================================================================ #
# LOOP
# ============================================================================ #

kid_cnt = 0
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
            
        # load KID data
        I, Q, targ = loadKIDData(dir_targ, roach, kid)

        # align the KID data
        I_align, Q_align = alignIQ(I, Q, dat_align_indices)

        # slice the KID data for this map
        I_slice, Q_slice = sliceIQ(I_align, Q_align, slice_i, slice_f)

        # free up memory
        del(I, Q, I_align, Q_align)

        # amplitude data
        A  = createAmp(I_slice, Q_slice)

        # phase data
        P  = createPhase(I_slice, Q_slice)

        # delta frequency data
        DF = createΔfx_grad(I_slice, Q_slice, *targ)

        # build the pixel values array for the map
        zz_A, zz_P, zz_DF = buildSingleKIDMaps(A, P, DF, ra_indices, dec_indices, zz_nan)
        # zz_A, zz_P, zz_DF = buildSingleKIDMaps(
        #     dat_slice['ra'], dat_slice['dec'], rr, dd, ra_bin, dec_bin, A, P, DF, rr_bins, dd_bins, ra_indices, dec_indices)

        # normalize the map values
        # we lose all absolute reference, but needed to combine
        # we don't need to keep the originals in memory so overwrite
        zz_A  = normImage(zz_A)
        zz_P  = normImage(zz_P)
        zz_DF = normImage(zz_DF)

        # some have inverted values for unknown reasons
        # so attempt to invert if necessary so source contains max val
        zz_A  = invertImageIfNeeded(zz_A)
        zz_P  = invertImageIfNeeded(zz_P)
        zz_DF = invertImageIfNeeded(zz_DF)

        # use first kid we process as the reference for alignment etc.
        # if kid_ref is None: kid_ref = kid
        if zz_A_ref is None:  zz_A_ref = zz_A
        if zz_P_ref is None:  zz_P_ref = zz_P
        if zz_DF_ref is None: zz_DF_ref = zz_DF
        
        log.info(f"ref KID = {kid_ref}")

        # calculate shift for each map from reference
        shift_A_bp  = imageShiftToAlign(zz_A_ref, zz_A, brightestPixel)
        shift_P_bp  = imageShiftToAlign(zz_P_ref, zz_P, brightestPixel)
        shift_DF_bp = imageShiftToAlign(zz_DF_ref, zz_DF, brightestPixel)
        shift_A_g  = imageShiftToAlign(zz_A_ref, zz_A, fit2DGaussian)
        shift_P_g  = imageShiftToAlign(zz_P_ref, zz_P, fit2DGaussian)
        shift_DF_g = imageShiftToAlign(zz_DF_ref, zz_DF, fit2DGaussian)

        # use the median shift (we have 3 so hopefully at least 2 are the same)
        # this is necessary because sometimes the source isn't max val in one map
        # shift_to_use = np.nanmedian([shift_A, shift_P, shift_DF], axis=0)

        # use the mean shift, but only for points close
        shift_to_use = meanPoint([shift_A_bp, shift_P_bp, shift_DF_bp, shift_A_g, shift_P_g, shift_DF_g])

        log.info(f"shift_A_bp={shift_A_bp}, shift_P_bp={shift_P_bp}, shift_DF_bp={shift_DF_bp}")
        log.info(f"shift_A_g={shift_A_g}, shift_P_g={shift_P_g}, shift_DF_g={shift_DF_g}")
        log.info(f"shift_to_use={shift_to_use}")

        # if no shifts are similar, skip this KID
        if shift_to_use is None:
            log.info(f"No concensus on shift to use - skipping this KID.")
            print("o", end="", flush=True)
            continue

        # align the images with chosen shift
        zz_A  = shift(zz_A, shift_to_use, cval=np.nan, order=0)
        zz_P  = shift(zz_P, shift_to_use, cval=np.nan, order=0)
        zz_DF = shift(zz_DF, shift_to_use, cval=np.nan, order=0)
        
        # add to combined maps, or create if needed  
        zz_A_multi  = ternNansum(zz_A_multi, zz_A) 
        zz_P_multi  = ternNansum(zz_P_multi, zz_P)
        zz_DF_multi = ternNansum(zz_DF_multi, zz_DF)
        
        # track how many kids had values for each pixel, for mean
        # zz_A_kid_cnt  = ternSum(zz_A_kid_cnt, ~np.isnan(zz_A)) 
        # zz_P_kid_cnt  = ternSum(zz_P_kid_cnt, ~np.isnan(zz_P))
        # zz_DF_kid_cnt = ternSum(zz_DF_kid_cnt, ~np.isnan(zz_DF))
            
        # number of KIDs processed
        kid_cnt += 1
            
        # output final and certain intermediary products
        if kid_cnt in outputAtKidCnt or kid_cnt == len(kids):
            
            # divide sum maps by count to get mean maps for output
            # zz_A_out  = np.divide(zz_A_multi, zz_A_kid_cnt, where=zz_A_kid_cnt.astype(bool))
            # zz_P_out  = np.divide(zz_P_multi, zz_P_kid_cnt, where=zz_P_kid_cnt.astype(bool))
            # zz_DF_out = np.divide(zz_DF_multi, zz_DF_kid_cnt, where=zz_DF_kid_cnt.astype(bool))
            zz_A_out  = zz_A_multi/kid_cnt
            zz_P_out  = zz_P_multi/kid_cnt
            zz_DF_out = zz_DF_multi/kid_cnt
            
            # hack to add nans back in
            # I can't figure out why this is needed
            # nans are lost in the divide
            # zz_A_out[zz_A_kid_cnt==0] = np.nan
            # zz_P_out[zz_P_kid_cnt==0] = np.nan
            # zz_DF_out[zz_DF_kid_cnt==0] = np.nan
            
            # generate output arrays
            A_out  = np.array([rr, dd, zz_A_out])
            P_out  = np.array([rr, dd, zz_P_out])
            DF_out = np.array([rr, dd, zz_DF_out])
            
            # generate filenames
            file_A  = os.path.join(dir_prods, f"map_A_{kid_cnt}_kids")
            file_P  = os.path.join(dir_prods, f"map_P_{kid_cnt}_kids")
            file_DF = os.path.join(dir_prods, f"map_DF_{kid_cnt}_kids")
            
            # save maps
            np.save(file_A, A_out)
            np.save(file_P, P_out)
            np.save(file_DF, DF_out)
            
            # delete intermediary vars from memory
            # if kid_cnt != len(kids):
            #     del(zz_A_out, zz_P_out, zz_DF_out, A_out, P_out, DF_out, file_A, file_P, file_DF)
            
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
# POST-LOOP
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

print("Done.")