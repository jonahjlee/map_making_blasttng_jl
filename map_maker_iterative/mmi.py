# ============================================================================ #
# mmi.py
#
# James Burgoyne jburgoyne@phas.ubc.ca 
# CCAT Prime 2024
#
# Map Maker Iterative. 
# The main naive iterative map maker script for BLAST-TNG data.
# ============================================================================ #


import os
import gc
import time
from datetime import datetime
import logging
import tracemalloc
import numpy as np

from mmi_config import *
import mmi_data_lib as dlib
# import mmi_tod_lib as tlib
import mmi_map_lib as mlib



# ============================================================================ #
# MAIN
# ============================================================================ #

def main():

    print(f"Creating maps for roach {roach}.")


# ============================================================================ #
#  M SETUP
# Setup administrative tasks, e.g. log, timing, memory tracking

    # memory tracking
    tracemalloc.start()

    # time tracking
    timer = Timer()

    # output directory
    dir_out = genDirsForRun(genTimestamp(), [dir_xform, dir_single])
    print(f"Output is in {dir_out}")

    # log file
    global log
    log = genLog(log_file, dir_out)

    # apply log decorator to libraries
    logLibFuncs([dlib, mlib])

    # Log the names and values of all configuration variables
    for var in [var for var in dir() if not var.startswith('__')]:
        value = globals()[var]
        log.info(f"{var} = {value}")


# ============================================================================ #
#  M LOAD
# Load and calibrate data common to all KIDs

    # load master data 
    dat_raw = dlib.loadMasterData(roach, dir_master, dir_roach)

    # load all target sweeps
    dat_targs, Ff = dlib.loadTargSweepsData(dir_targ)

    # temporaly align tods, rebin if necessary
    dat_aligned, dat_align_indices = dlib.alignMasterAndRoachTods(dat_raw)

    # slice tods to desired region (remove cal lamp)
    dat_sliced = {
        field: dat_aligned[field][slice_i:cal_i].copy() 
        for field in dat_aligned}

    # free memory and force collection (these tods are large)
    del(dat_raw, dat_aligned)
    gc.collect()


# ============================================================================ #
#  M COORDS
# Map coordinates and axis arrays

    # source coordinates in az/el telescope frame
    source_azel = mlib.sourceCoordsAzEl( # source_azel.az, source_azel.el
        source_name, 
        dat_sliced['lat'], dat_sliced['lon'], 
        dat_sliced['alt'], dat_sliced['time'])

    # generate x and y, the az/el offset tods
    x, y = mlib.azElOffsets(source_azel, dat_sliced['az'], dat_sliced['el'])

    # convert offsets in degrees to um on image plane
    x, y = mlib.offsetsTanProj(x, y, platescale)

    # generate map bins and axes
    xx, yy, x_bins, y_bins, x_edges, y_edges = mlib.genMapAxesAndBins(
        x, y, x_bin, y_bin)


# ============================================================================ #
#  M KIDs
# Determine which KIDs to use

    # kids to use
    kids = dlib.findAllKIDs(dir_roach) # all in dir_roach; sorted

    # remove unused channels
    kids = [kid for kid in kids if int(kid) >= kid_max]

    # KID rejects
    try: # file might not exist
        kid_rejects = dlib.loadKidRejects(file_rejects)
        kids = [kid for kid in kids if kid not in kid_rejects]
    except: pass
        
    # move ref kid so it's processed first
    # this is last so it raises an error if our ref has been removed
    kids.remove(kid_ref)
    kids.insert(0, kid_ref)


# ============================================================================ #
#  M COM LOOP
# The common-mode iterative refinement loop

    combined_map = None

    for iteration in range(10): # currently just doing 10 loops for testing
        # is there a metric to decide when to stop looping?

        # create dir for this iteration
        dir_it = os.path.join(dir_out, f'it_{iteration}')
        os.makedirs(dir_it, exist_ok=True)

        # common mode KID loop
        # loop over KIDs, generate common mode
        common_mode = mlib.commomModeLoop(
            kids, dat_targs, Ff, dat_align_indices, 
            roach, dir_roach, slice_i, cal_i, cal_f, 
            x, y, x_edges, y_edges, source_azel, combined_map)
        np.save(os.path.join(dir_it, file_commonmode), common_mode)

        # combine maps loop
        # loop over KIDs, generate single maps, combine
        def save_singles_func(kid, data):
            # we can probably return all single maps and then save here?
            np.save(os.path.join(dir_it, dir_single, f"map_kid_{kid}"), data)
        combined_map, shifts, source_azel = mlib.combineMapsLoop(
            kids, dat_targs, Ff, dat_align_indices, roach, dir_roach, 
            slice_i, cal_i, cal_f, x, y, x_edges, y_edges, xx, yy, common_mode,
            save_singles_func)

        # output combined map to file
        if dir_out is not None:
            np.save(os.path.join(dir_it, f"combined_map"), 
                    [xx, yy, combined_map])

        # save shifts to file
        file_shifts = os.path.join(dir_it, dir_xform, f'shifts.npy')
        np.save(file_shifts, shifts)




# ============================================================================ #
# FUNCTIONS
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
# logFunction
def logFunction(func, printMessage=None):
    '''Decorator to log function details and print message.
    '''

    def wrapper(*args, **kwargs):
        if printMessage: 
            print(printMessage)
            log.info(printMessage)
        log.info(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        log.info(f"Finished {func.__name__}")
        return result
    
    return wrapper


# ============================================================================ #
# logLibFuncs
def logLibFuncs(libs):
    '''Modify library funcs to add log decorator.

    libs: (list) Libraries to modify.
    '''

    for lib in libs:
        # Iterate over the functions in the library
        for name in dir(lib):
            func = getattr(lib, name)
            # Check if the function has the marker decorator applied
            if hasattr(func, 'logThis') and func.logThis:
                # Apply the log decorator to the function
                setattr(lib, name, logFunction(func))


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




if __name__ == "__main__":
    main()
