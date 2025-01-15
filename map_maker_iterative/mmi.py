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
    dir_out = makeBaseDir()
    print(f"Output is in {dir_out}")

    # log file
    global log
    log = genLog(log_file, dir_out)

    # apply log decorator to libraries
    logLibFuncs([dlib, mlib])

    # Log the names and values of all configuration variables
    combined_scope = {**globals(), **locals()}
    for key, value in combined_scope.items():
        log.info(f"{key} = {value}")
    # for var in [var for var in dir() if not var.startswith('__')]:


# ============================================================================ #
#  M LOAD
# Load and calibrate data common to all KIDs

    print(f"Loading common data... ", end="", flush=True)

    # load master data 
    dat_raw = dlib.loadMasterData(roach, dir_master, dir_roach)

    # load all target sweeps
    dat_targs, Ff = dlib.loadTargSweepsData(dir_targ)

    # load array layout and pre-calculate map shifts from layout
    dat_layout = dlib.abFromLayout(file_layout)
    shifts_xy_layout = mlib.xyFromAb(dat_layout, platescale, pixels_per_beam, psf)

    # temporaly align tods, rebin if necessary
    dat_aligned, dat_align_indices = dlib.alignMasterAndRoachTods(dat_raw)

    # calculate sampling frequency
    fs_tod = dlib.samplingFrequency(dat_aligned['time'])

    # calculate spatial bin diff.
    ds_tod = dlib.ds(dat_aligned['az'], dat_aligned['el'])
    
    # high pass filter cutoff frequency
    fc_high = mlib.cutoffFrequency(fc_high_scale, 1/fs_tod, ds_tod)

    # print(f"fs_tod={fs_tod}")
    # print(f"ds_tod={ds_tod}")
    # print(f"fc_high_scale={fc_high_scale}")
    # print(f"fc_high={fc_high}")
    # exit()

    # slice tods to desired region (remove cal lamp)
    dat_sliced = {
        field: dat_aligned[field][slice_i:cal_i].copy() 
        for field in dat_aligned}

    # free memory and force collection (these tods are large)
    del(dat_raw, dat_aligned)
    gc.collect()

    print("Done.")


# ============================================================================ #
#  M COORDS
# Map coordinates and axis arrays

    print(f"Building map base and coordinates... ", end="", flush=True)

    # detected source coordinates in az/el telescope frame
    source_azel = mlib.sourceCoordsAzEl( # (az, el)
        source_name, 
        dat_sliced['lat'], dat_sliced['lon'], 
        dat_sliced['alt'], dat_sliced['time'])

    # generate x_az and y_el, the az/el offset tods
    x_az, y_el = mlib.azElOffsets(source_azel, dat_sliced['az'], dat_sliced['el'])

    # convert offsets in degrees to um on image plane
    x_um, y_um = mlib.offsetsTanProj(x_az, y_el, platescale)

    # generate map bins and axes
    xx, yy, x_bins, y_bins, x_edges, y_edges = mlib.genMapAxesAndBins(
        x_um, y_um, x_bin, y_bin)
    
    print("Done.")


# ============================================================================ #
#  M KIDs
# Determine which KIDs to use

    print(f"Loading KID data... ", end="", flush=True)

    # kids to use
    kids = dlib.findAllKIDs(dir_roach) # all in dir_roach; sorted

    # remove unused channels
    kids = [kid for kid in kids if int(kid) <= kid_max]

    # KID rejects
    try: # file might not exist
        kid_rejects = dlib.loadKidRejects(file_rejects)
        kids = [kid for kid in kids if kid not in kid_rejects]
    except: pass

    # remove kids not in layout file
    kids = [kid for kid in kids if kid in shifts_xy_layout.keys()]

    # move ref kid so it's processed first
    # this is last so it raises an error if our ref has been removed
    kids.remove(kid_ref)
    kids.insert(0, kid_ref)

    print("Done.")


# ============================================================================ #
#  M COM LOOP
# The common-mode iterative refinement loop

    print(f"Performing {ct_its} common-mode iterations: ", end="", flush=True)

    combined_map = None
    source_xy = None

    for iteration in range(ct_its):

        print(f"{iteration+1} ", end="", flush=True)

        # create dir and subdirs for this iteration
        dir_it = os.path.join(dir_out, f'it_{iteration}')
        makeDirs([dir_single, dir_xform], dir_it)

        # common mode KID loop
        # loop over KIDs, generate common mode
        common_mode = mlib.commomModeLoop(
            kids, dat_targs, Ff, dat_align_indices, 
            roach, dir_roach, slice_i, cal_i, cal_f, 
            x_um, y_um, x_edges, y_edges, source_xy, combined_map,
            fs_tod, fc_high)
        np.save(os.path.join(dir_it, file_commonmode), common_mode)

        # combine maps loop
        # loop over KIDs, generate single maps, combine
        def save_singles_func(kid, data):
            # we can probably return all single maps and then save here?
            np.save(os.path.join(dir_it, dir_single, f"map_kid_{kid}"), data)
        combined_map, shifts_source, source_xy = mlib.combineMapsLoop(
            kids, dat_targs, Ff, dat_align_indices, roach, dir_roach, 
            slice_i, cal_i, cal_f, x_um, y_um, x_edges, y_edges, xx, yy, common_mode,
            fs_tod, fc_high, save_singles_func, 
            None)
            # shifts_xy_layout)

        # output combined map to file
        if dir_out is not None:
            np.save(os.path.join(dir_it, f"combined_map"), 
                    [xx, yy, combined_map])

        # save shifts to file
        np.save(os.path.join(dir_it, dir_xform, f'shifts_source.npy'), 
                shifts_source)
        np.save(os.path.join(dir_it, dir_xform, f'shifts_xy_layout.npy'), 
                shifts_xy_layout)
        

    print("Done.")
    print(f"Time taken: {timer.deltat()}")




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
# makeBaseDir
def makeBaseDir():
    '''Make base directory for this run.
    '''
    
    # use timestamp as unique dir suffix
    suffix = genTimestamp()

    # base directory
    dir_base = os.path.join(os.getcwd(), f"map_{suffix}")
    os.makedirs(dir_base)

    return dir_base
    

# ============================================================================ #
# makeDirs
def makeDirs(dirs, dir_base=None):
    '''Generate requested directories.

    dir_base: Root folder to create directories in.
    dirs: List of directory names.
    '''

    for d in dirs:
        dir = d if dir_base is None else os.path.join(dir_base, d)
        os.makedirs(dir, exist_ok=True)




if __name__ == "__main__":
    main()
