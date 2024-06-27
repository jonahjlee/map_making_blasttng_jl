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

import mmi_data_lib as dlib


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
# getCalKidDf
def getCalKidDf(kid, dat_targs, Ff, dat_align_indices, 
                roach, dir_roach, i_i, i_cal, i_f):
    '''Load, calculate, and calibrate the df tod for a KID.

    kid: (str) The KID number, e.g. '0001'.
    dat_targs: (tuple of 2 1D arrays of floats) Target sweep I and Q.
    Ff: (1D array of floats) Frequency axis for target sweep.
    dat_align_indices: (list of ints) Indices to align tods.
    roach: (int) The roach number.
    dir_roach: (string) The directory that roach data is in.
    i_i: (int) Desired data first index.
    i_cal: (int) First index of calibration lamp data.
    i_f: (int) Final index.
    '''

    # load I and Q (memmap)
    I, Q = dlib.loadKIDData(roach, kid, dir_roach)

    # load target sweep
    targ = dlib.getTargSweepIQ(kid, dat_targs)

    # slice and align (include cal lamp in slice for now)
    I_slice = I[dat_align_indices[i_i:i_f]] # slicing align indices
    Q_slice = Q[dat_align_indices[i_i:i_f]]

    # build df tod
    tod = df_IQangle(I_slice, Q_slice, *targ, Ff)

    # normalize tod data to calibration lamp
    tod = normTod(tod, i_cal - i_i, i_f - i_i)

    # slice away calibration region
    # brings master and roach tods in sync
    tod = tod[:i_cal - i_i]

    return tod