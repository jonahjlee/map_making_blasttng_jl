# ============================================================================ #
# data_loader.py
#
# Jonah Lee
#
# Data Loader
# Load BLAST-TNG data from the HAA CCAT Control Computer into .npy files
# for use in kid_viewer.py. Intended use is to
# ============================================================================ #

import sys
import os
import numpy as np
# modify PATH environment variable to access modules in other directory
sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path.append(os.path.join(os.getcwd(), '..', 'map_maker_iterative'))
from map_maker_iterative.mmi_config import RoachID, ScanPass
from map_maker_iterative.mmi_roach import RoachPass


def get_norm_df_dict(roach):
    return {kid:roach.get_norm_kid_df(kid) for kid in roach.kids}

def get_df_dict(roach):
    return {kid:roach.get_kid_df(kid) for kid in roach.kids}


def downsample(arr: np.ndarray, factor, allow_truncate=False):
    assert arr.ndim == 1, "can only down-sample 1-d array"
    if allow_truncate: arr = arr[:-(arr.size % factor)]
    else: assert arr.size % factor == 0, "array length must be a multiple of down-sampling factor"
    reshaped = np.reshape(arr, (-1, factor))
    return reshaped.mean(axis=1)

def apply_to_values(mydict: dict, func: callable, *args, **kwargs):
    """passes the dict value as the first argument to func"""
    return {key:func(val, *args, **kwargs)for key, val in mydict.items()}

def process_npy_files(root_folder):
    # Walk through all the directories and subdirectories
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.npy') and not file.endswith('um.npy'):
                # Construct the full file path
                file_path = os.path.join(root, file)


                breakpoint()

                # Load the dictionary from the .npy file
                data = np.load(file_path, allow_pickle=True).item()

                # Create a new dictionary by applying downsample to each key's value
                stringkey_data = {f'{key:04}': value for key, value in data.items()}

                breakpoint()

                # Save the new dictionary to the new file
                np.save(file_path, stringkey_data)


if __name__ == '__main__':

    print('\nLoading RoachPass...')
    roach = RoachPass(RoachID(1), ScanPass.ALL, use_rejects_file=False)
    print(roach.info)

    out_dir = os.path.join(os.getcwd(), 'data', f'roach_{roach.id}_{roach.scan_pass.name.lower()}')
    os.makedirs(out_dir, exist_ok=True)
    print('Created output directory: ', out_dir)

    # norm_df_file = os.path.join(out_dir, 'norm_df_dict_ds_10')
    # ds_by_10 = apply_to_values(get_norm_df_dict(roach), downsample, 10)
    # np.save(norm_df_file, ds_by_10)
    # print('\nSaved normalized DF dict to: ', norm_df_file)

    x_um_file = os.path.join(out_dir, 'x_um.npy')
    y_um_file = os.path.join(out_dir, 'y_um.npy')
    np.save(roach.x_um, x_um_file)
    np.save(roach.y_um, y_um_file)