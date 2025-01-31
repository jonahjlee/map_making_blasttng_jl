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
# import mmi_tod_lib as tlib
import mmi_map_lib as mlib


# ============================================================================ #
# MAIN
# ============================================================================ #

def main():

    print(f"Creating maps for roaches {roaches}.")


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

    roach_data = {}
    for roach in roaches:
        roach_data[roach] = {}

    for roach, data in roach_data.items():

        if pass_to_map == ScanPass.ALL:
            data['slice_i'] = slice_i_dict[roach]
            data['slice_f'] = data['slice_i'] + pass_indices[3]

        else:
            data['slice_i'] = slice_i_dict[roach] + pass_indices[pass_to_map.value]
            data['slice_f'] = data['slice_i'] + pass_indices[pass_to_map.value + 1]

        data['cal_i'] = slice_i_dict[roach] + cal_i_offset
        data['cal_f'] = slice_i_dict[roach] + cal_f_offset
        data['dir_roach'] = dir_roach_dict[roach]
        data['dir_targ'] = dir_targ_dict[roach]

        data['dat_targs'], data['Ff'], data['dat_align_indices'], data['dat_sliced'] = dlib.loadSlicedData(
            roach,
            data['slice_i'],
            data['cal_i'],
            dir_master,
            data['dir_roach'],
            data['dir_targ'])


    print("Done.")


# ============================================================================ #
#  M COORDS
# Map coordinates and axis arrays

    print(f"Building map base and coordinates... ", end="", flush=True)

    for roach, data in roach_data.items():
        # detected source coordinates in az/el telescope frame
        source_azel = mlib.sourceCoordsAzEl( # (az, el)
            source_name,
            data['dat_sliced']['lat'], data['dat_sliced']['lon'],
            data['dat_sliced']['alt'], data['dat_sliced']['time'])

        # generate x_az and y_el, the az/el offset tods
        data['x_az'], data['y_el'] = mlib.azElOffsets(
            source_azel,
            data['dat_sliced']['az'],
            data['dat_sliced']['el']
        )

        # convert offsets in degrees to um on image plane
        data['x_um'], data['y_um'] = mlib.offsetsTanProj(data['x_az'], data['y_el'], platescale)

    # generate map bins and axes
    xx, yy, x_bins, y_bins, x_edges, y_edges \
        = mlib.genMapAxesAndBins(roach_data, x_bin, y_bin)

    print("Done.")


# ============================================================================ #
#  M KIDs
# Determine which KIDs to use

    print(f"Loading KID data... ", end="", flush=True)

    for roach, data in roach_data.items():

        # kids to use
        kids = dlib.findAllKIDs(dir_roach_dict[roach]) # all in dir_roach; sorted

        # remove unused channels
        kids = [kid for kid in kids if int(kid) <= kid_max_dict[roach]]

        # KID rejects
        try: # file might not exist
            kid_rejects = dlib.loadKidRejects(file_rejects_dict[roach])
            kids = [kid for kid in kids if kid not in kid_rejects]
        except FileNotFoundError: pass

        # move ref kid so it's processed first
        # this is last so it raises an error if our ref has been removed
        kids.remove(kid_ref_dict[roach])
        kids.insert(0, kid_ref_dict[roach])

        data['kids'] = kids

    print("Done.")

    breakpoint()

# ============================================================================ #
#  M COM LOOP
# The common-mode iterative refinement loop

    print("Generating naive map")

    # create dir and subdirs for this iteration
    log.info("Note: iteration numbering has been updated such that it_0 contains no common-mode filtering,")
    log.info("      it_1 contains one pass of filtering, etc.")
    dir_it = os.path.join(dir_out, f'it_0')
    makeDirs([dir_single, dir_xform], dir_it)

    # start without common mode estimate
    for roach, data in roach_data.items():
        data['common_mode'] = 0

    # combine maps loop
    # loop over KIDs, generate single maps, combine
    def save_singles_func(kid, data):
        # we can probably return all single maps and then save here?
        np.save(os.path.join(dir_it, dir_single, f"map_kid_{kid}"), data)
    combined_map, shifts_source, source_xy = mlib.combineMapsLoop(
        roach_data, xx, yy, x_edges, y_edges, save_singles_func)

    # output combined map to file
    if dir_out is not None:
        np.save(os.path.join(dir_it, f"combined_map"),
                [xx, yy, combined_map])

    print(f"Performing {ct_its} common-mode iterations: ")

    for iteration in range(1, ct_its + 1):

        print(f"Starting iteration {iteration}.")

        # create dir and subdirs for this iteration
        dir_it = os.path.join(dir_out, f'it_{iteration}')
        makeDirs([dir_single, dir_xform], dir_it)

        # common mode KID loop
        # loop over KIDs, generate common mode

        # use a different common_mode for each roach
        for roach, data in roach_data.items():
            data['common_mode'] = mlib.commonModeLoop(roach, data, x_edges, y_edges, source_xy, combined_map)
            np.save(os.path.join(dir_it, file_commonmode(roach)), data['common_mode'])

        combined_map, shifts_source, source_xy = mlib.combineMapsLoop(
            roach_data, xx, yy, x_edges, y_edges, save_singles_func)

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
