# ============================================================================ #
# mmi_tod_lib.py
#
# James Burgoyne jburgoyne@phas.ubc.ca 
# CCAT Prime 2024
#
# Map Maker Iterative time-ordered data (tod) library. 
# ============================================================================ #


import sys
import warnings
import numpy as np
# from scipy.fft import rfft, irfft, rfftfreq
import scipy as sp

import mmi_data_lib as dlib


# ============================================================================ #
# df_IQangle (updated for sub-loops)
"""
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
    f0 = Ff[i_f0]
    
    cI = (If.max() + If.min())/2            # centre of target IQ loop
    cQ = (Qf.max() + Qf.min())/2
    
    def normalizeAngles(a, m): # +/-π around m
        return ((a - m) + np.pi) % (2*np.pi) - np.pi + m
    
    # observations
    I_c, Q_c = I - cI, Q - cQ               # shift origin
    θ = np.arctan2(Q_c, I_c)                # find IQ angles
    mean_θ = np.mean(θ)                     # mean observation angle
    θ = normalizeAngles(θ, mean_θ)          # to cont. angles around mean
    
    # target sweep 
    If_c, Qf_c = If - cI, Qf - cQ           # shift center to origin
    θf = np.arctan2(Qf_c, If_c)             # find IQ angles
    θf = normalizeAngles(θf, mean_θ)        # to cont. angles around obs. mean
    
    # is one-to-many?
    diffs_θf = np.diff(θf)
    onetoone = np.all(diffs_θf > 0) or np.all(diffs_θf < 0)
    
    # trim sweep to obs. range
    mask_obs_range = (θf > θ.min()) & (θf < θ.max()) 
    θf = θf[mask_obs_range]
    Ff = Ff[mask_obs_range]
    
    # this will potentially have one-to-many from loops
    # find best contiguous region
    if not onetoone:
        diffs_Ff = np.diff(Ff)
        split_indices = np.where(diffs_Ff > 2*np.median(diffs_Ff))[0] + 1
        regions_θf = np.split(θf, split_indices)
        regions_Ff = np.split(Ff, split_indices)
        i_best = None # region that contains obs. and has largest mean angle
        for i in range(len(regions_θf)):  
            ex = (regions_θf[i][0], regions_θf[i][-1]) # don't know if asc/des
            if min(ex) < mean_θ < max(ex):
                if i_best is None or np.mean(regions_θf[i]) > np.mean(regions_θf[i_best]):
                    i_best = i
        if i_best is not None:
            θf = regions_θf[i_best]
            Ff = regions_Ff[i_best]
    
    # np.interp requires ascending
    if θf[0] > θf[-1]:                      
        θf = θf[::-1]
        Ff = Ff[::-1]
    
    # interpolate
    df = np.interp(θ, θf, Ff - f0) # using df from resonance
    
    return df/f0
"""


# ============================================================================ #
# normTod
def downsample(arr: np.ndarray, factor):
    assert arr.ndim == 1, "can only down-sample 1-d array"
    assert arr.size % factor == 0, "array length must be a multiple of down-sampling factor"
    reshaped = np.reshape(arr, (-1, factor))
    return reshaped.mean(axis=1)


# ============================================================================ #
# df_IQangle
# """
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
# """


# ============================================================================ #
# df_IQangle (gradient method)
"""
def df_IQangle(I, Q, If, Qf, Ff, i_f0=None): 
    dIfdf = np.diff(If)/1e3        # Δx is const. so Δy=dI/df
    dQfdf = np.diff(Qf)/1e3        # /1e3 for units
    dIfdff0 = dIfdf[len(dIfdf)//2] # dI(f)/df at f0
    dQfdff0 = dQfdf[len(dQfdf)//2] # assume f0 is centre index
    I_n = I - np.mean(I)           # centre values on 0
    Q_n = Q - np.mean(Q)           #
    
    den = dIfdff0**2 + dQfdff0**2  # 
    
    numx = ((I_n*dIfdff0 + Q_n*dQfdff0))
    Δfx = numx/den
    
    # numy = ((Q_n*dIfdff0 - I_n*dQfdff0))
    # Δfy = numy/den
    
    return Δfx/Ff[len(dQfdf)//2]
"""


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
# getNormKidDf
def getNormKidDf(kid, dat_targs, Ff, dat_align_indices, 
                roach, dir_roach, slice_i, slice_f, cal_i_offset, cal_f_offset):
    '''Load, calculate, and calibrate the df tod for a KID.

    kid: (str) The KID number, e.g. '0001'.
    dat_targs: (tuple of 2 1D arrays of floats) Target sweep I and Q.
    Ff: (1D array of floats) Frequency axis for target sweep.
    dat_align_indices: (list of ints) Indices to align tods.
    roach: (int) The roach number.
    dir_roach: (string) The directory that roach data is in.
    slice_i: (int) Desired data first index (inclusive).
    slice_f: (int) Desired data last index (exclusive).
    cal_f_offset: (int) First index since slice_i of calibration lamp data (inclusive).
    cal_f_offset: (int) Last index since slice_i of calibration lamp data (exclusive).
    '''

    # load I and Q (memmap)
    I, Q = dlib.loadKIDData(roach, kid, dir_roach)

    # load target sweep
    targ = dlib.getTargSweepIQ(kid, dat_targs)

    # slice and align (include cal lamp in slice for now)
    I_slice = I[dat_align_indices[slice_i:slice_i+cal_f_offset]]
    Q_slice = Q[dat_align_indices[slice_i:slice_i+cal_f_offset]]

    # build df tod
    tod = df_IQangle(I_slice, Q_slice, *targ, Ff)

    # normalize tod data to calibration lamp
    tod = normTod(tod, cal_i_offset, cal_f_offset)

    # slice only desired region
    # brings master and roach tods in sync
    tod = tod[:slice_f - slice_i]

    return tod


# ============================================================================ #
# removeTodOutliers
def removeTodOutliers(tod, sigmas=3.5, n=2):
    '''Remove outliers in the tod, e.g. cosmic ray strikes.

    tod: (1D array of floats) The time-ordered data.
    sigmas: (float) The multiples of sigma 1-bin increase to filter at.
    n: (float) Number of bins on either side to replace.
    '''
    
    # avoid altering original
    # tod0 = tod
    tod = tod.copy()
    
    # tod bin-to-bin diffs
    dtod = np.diff(tod, append=tod[-1]) # repeat last element to pad
    
    # find 1 bin changes > some multiple of sigma (std)
    indices = np.where(dtod > np.std(tod)*sigmas)[0] + 1 
    
    if len(indices) > 0:

        # Create a mask for all indices that need to be interpolated
        mask = np.zeros_like(tod, dtype=bool)
        for i in indices:
            start = max(0, i - n)
            end = min(len(tod) - 1, i + n)
            if end > start:
                mask[start:end + 1] = True
        
        # Interpolate over the entire array using the valid points
        valid_indices = np.where(~mask)[0] # points to interpolate
        interpolated_values = np.interp(
            np.arange(len(tod)), valid_indices, tod[valid_indices])
        tod[mask] = interpolated_values[mask] # only replace valid points
        # may be able to gain efficiencies here
        # by not interpolating over entire array

    return tod


# ============================================================================ #
# fillTodGaps
def fillTodGaps(tod):
    ''''''

    # need to test for if gaps even exist

    return tod


# ============================================================================ #
# cleanTOD
def cleanTOD(tod):
    ''''''

    # remove outliers (e.g. cosmic rays)
    tod = removeTodOutliers(tod)

    # fill gaps
    tod = fillTodGaps(tod)

    # handle edges
    ## maybe?

    # smooth
    ## maybe?

    return tod


# ============================================================================ #
# highpassFilterTOD
def highpassFilterTOD(tod, sampling_rate, cutoff_freq):
    """Apply a high-pass filter using FFT.

    Parameters:
    - tod: numpy array, the time-ordered data to filter.
    - sampling_rate: float, the sampling rate of the data in Hz.
    - cutoff_freq: float, the cutoff frequency for the high-pass filter in Hz.

    Returns:
    - filtered_tod: numpy array, the filtered time-ordered data.
    """

    # FFT of the TOD
    tod_fft = sp.fft.rfft(tod)
    freqs = sp.fft.rfftfreq(len(tod), d=1/sampling_rate)

    # Create a high-pass filter mask
    filter_mask = freqs > cutoff_freq

    # Apply the filter in the frequency domain
    tod_fft_filtered = tod_fft * filter_mask

    # Inverse FFT to get back to time domain
    filtered_tod = sp.fft.irfft(tod_fft_filtered)

    return filtered_tod
