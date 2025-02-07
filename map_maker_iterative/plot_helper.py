# ============================================================================ #
# plot_helper.py
#
# Jonah Lee
#
# Plot Helper
# Quick tool to set up an environment where user can build and save plots
# on a remote connection
# ============================================================================ #


# ============================================================================ #
# IMPORTS
# ============================================================================ #

import os
import numpy as np
import matplotlib.pyplot as plt

from mmi_config import *
from mmi_roach import Roach
import mmi_data_lib as dlib
import mmi_tod_lib as tlib
import mmi_map_lib as mlib


# ============================================================================ #
# HELPER FUNCTIONS
# ============================================================================ #

out_dir = os.getcwd()

def save_plt(fname="tmp", title=None):
    plt.title(title if title is not None else fname)
    plt.savefig(os.path.join(out_dir, fname))
    plt.clf()

def plot_grid(data: list[np.ndarray], cols=3):
    rows = int(np.ceil(len(data) / cols))
    fig, axes = plt.subplots(rows, cols)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(data):
            ax.plot(data[i])
        else:
            ax.axis('off')
    plt.tight_layout()


# ============================================================================ #
# ENTRY POINT
# ============================================================================ #

if __name__ == '__main__':

    print(f"Loading common data... ")
    roaches = {}
    for roach_id in roach_ids:
        roaches[roach_id] = Roach(roach_id, pass_to_map)
        print(roaches[roach_id].info)
    print("Done.")

    print(f"Building map bins and axes... ", end="", flush=True)
    xx, yy, x_bins, y_bins, x_edges, y_edges \
        = mlib.genMapAxesAndBins(roaches.values(), x_bin, y_bin)
    print("Done.")

    # user now has access to `roaches` and can compute mapping, common mode, or kid DF

    print("Entering breakpoint!")
    breakpoint()
