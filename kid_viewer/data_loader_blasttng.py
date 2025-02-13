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
    return {int(kid):roach.get_norm_kid_df(kid) for kid in roach.kids}

def get_df_dict(roach):
    return {int(kid):roach.get_kid_df(kid) for kid in roach.kids}


if __name__ == '__main__':

    print('\nLoading RoachPass...')
    roach = RoachPass(RoachID(1), ScanPass.ALL, use_rejects_file=False)
    print(roach.info)

    out_dir = os.path.join(os.getcwd(), 'data', f'roach_{roach.id}_{roach.scan_pass.name.lower()}')
    print('Created output directory: ', out_dir)

    norm_df_file = os.path.join(out_dir, 'norm_df_dict.npy')
    np.save(norm_df_file, get_norm_df_dict(roach))
    print('\nSaved normalized DF dict to: ', norm_df_file)
