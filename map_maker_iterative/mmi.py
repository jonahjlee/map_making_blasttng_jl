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
import time
from datetime import datetime
import logging
import tracemalloc
import numpy as np

from mmi_config import *
import mmi_data_lib as dlib
from mmi_roach import Roach
# import mmi_tod_lib as tlib
import mmi_map_lib as mlib


# ============================================================================ #
# MAIN
# ============================================================================ #

def main():

    print(f"Creating maps for {[r.name for r in roach_ids]}.")


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

    print(f"Loading common data... ")

    roaches = {}

    for roach_id in roach_ids:
        roaches[roach_id] = Roach(roach_id, pass_to_map)
        log.info(roaches[roach_id].info)
        print(roaches[roach_id].info)

    # trigger common data loading
    # for roach in roaches.values():
    #     roach._load_master_data()
    #     roach._load_target_sweeps()

    print("Done.")


# ============================================================================ #
#  M COORDS
# Map coordinates and axis arrays

    print(f"Building map base... ", end="", flush=True)

    # generate map bins and axes
    xx, yy, x_bins, y_bins, x_edges, y_edges \
        = mlib.genMapAxesAndBins(roaches.values(), x_bin, y_bin)

    print("Done.")


# ============================================================================ #
#  M KIDs
# Determine which KIDs to use

    print(f"Loading KID data... ", end="", flush=True)

    # moved into Roach.__init__ / Roach._load_kids()

    print("Done.")

    # breakpoint()

# ============================================================================ #
#  M COM LOOP
# The common-mode iterative refinement loop

    print("Generating naive map")

    # create dir and subdirs for this iteration
    log.info("Note: iteration numbering has been updated such that it_0 contains no common-mode filtering,")
    log.info("      it_1 contains one pass of filtering, etc.")
    dir_it = os.path.join(dir_out, f'it_0')
    makeDirs([dir_single, dir_xform], dir_it)

    # combine maps loop
    # loop over KIDs, generate single maps, combine
    def save_singles_func(kid, data):
        # we can probably return all single maps and then save here?
        np.save(os.path.join(dir_it, dir_single, f"map_kid_{kid}"), data)

    combined_map, shifts_source, source_xy = mlib.combineMapsLoop(
        roaches.values(),
        cal_i_offset, cal_f_offset,
        xx, yy,
        x_edges, y_edges,
        0,
        down_sample_factor,
        save_singles_func
    )

    # output combined map to file
    if dir_out is not None:
        np.save(os.path.join(dir_it, "combined_map"),
                [xx, yy, combined_map])

    print(f"Performing {ct_its} common-mode iterations: ")

    for iteration in range(1, ct_its + 1):

        print(f"Starting iteration {iteration}.")

        # create dir and subdirs for this iteration
        dir_it = os.path.join(dir_out, f'it_{iteration}')
        makeDirs([dir_single, dir_xform], dir_it)

        # common mode KID loop
        # loop over KIDs, generate common mode
        common_mode = mlib.commonModeLoop(
            roaches.values(),
            cal_i_offset, cal_f_offset,
            x_edges, y_edges,
            source_xy,
            combined_map,
            down_sample_factor
        )
        np.save(os.path.join(dir_it, "common_mode"), common_mode)

        combined_map, shifts_source, source_xy = mlib.combineMapsLoop(
            roaches.values(),
            cal_i_offset, cal_f_offset,
            xx, yy,
            x_edges, y_edges,
            common_mode,
            down_sample_factor,
            save_singles_func
        )

        # output combined map to file
        if dir_out is not None:
            np.save(os.path.join(dir_it, f"combined_map"), 
                    [xx, yy, combined_map])

        # save shifts to file
        np.save(os.path.join(dir_it, dir_xform, f'shifts_source.npy'), 
                shifts_source)

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
