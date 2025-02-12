# ============================================================================ #
# kid_viewer.py
#
# Jonah Lee
#
# Kid Viewer
# Data visualization tool: display BLAST-TNG KID DF data as seen by the KID array.
# ============================================================================ #


# ============================================================================ #
# IMPORTS
# ============================================================================ #

import tkinter as tk
import numpy as np
import os
import pandas as pd
from scatter_animation import AnimatedScatterPlot


# ============================================================================ #
# HELPER FUNCTIONS
# ============================================================================ #

def load_kid_layout(layout_file, rejects_file=None):
    if layout_file.endswith('.csv'):
        return load_kid_layout_csv(layout_file, rejects_file)
    if layout_file.endswith('.npy'):
        return load_kid_layout_npy(layout_file, rejects_file)

    raise ValueError('Layout file must end with .csv or .npy')


def load_kid_layout_csv(layout_file, rejects_file=None) -> dict[int, tuple[float, float]]:
    """Loads KID x/y coordinates on the image plane for a ROACH

    Returns a dictionary which maps KID IDs to coordinate pairs.
    """
    try:
        df = pd.read_csv(layout_file, index_col=0)

        if rejects_file is not None:
            try:
                rejects = np.astype(
                    np.loadtxt(rejects_file, delimiter=' ', dtype=str),
                    int
                )
                df = df[~df.index.isin(rejects)]
            except FileNotFoundError as err:
                print(f"File {rejects_file} not found")
                raise err

        return {kid: (df['x'][kid], df['y'][kid]) for kid in df.index}

    except FileNotFoundError as err:
        print(f"File {layout_file} not found")
        raise err


def load_kid_layout_npy(layout_file, rejects_file=None) -> dict[int, tuple[float, float]]:
    """Loads KID x/y coordinates on the image plane for a ROACH

    Returns a dictionary which maps KID IDs to coordinate pairs.
    """
    try:
        layouts_dict = np.load(layout_file, allow_pickle=True).item()

        # parse keys in the form '0000' or 'roach1_0000'
        if isinstance(next(iter(layouts_dict.keys())), str):
            layouts_dict = {int(key[-4:]): val for key, val in layouts_dict.items() if key in layouts_dict}

        if rejects_file is not None:
            try:
                rejects = np.astype(
                    np.loadtxt(rejects_file, delimiter=' ', dtype=str),
                    int
                )
                layouts_dict = {key: val for key, val in layouts_dict.items() if key in rejects}
            except FileNotFoundError as err:
                print(f"File {rejects_file} not found")
                raise err

        return layouts_dict

    except FileNotFoundError as err:
        print(f"File {layout_file} not found")
        raise err


def downsample(arr: np.ndarray, factor, allow_truncate=False):
    assert arr.ndim == 1, "can only down-sample 1-d array"
    if allow_truncate: arr = arr[:-(arr.size % factor)]
    else: assert arr.size % factor == 0, "array length must be a multiple of down-sampling factor"
    reshaped = np.reshape(arr, (-1, factor))
    return reshaped.mean(axis=1)


# ============================================================================ #
# ENTRY POINT
# ============================================================================ #

if __name__ == '__main__':

    # Define Constants
    # layout_file = os.path.join(os.getcwd(), '..', 'detector_layouts', 'layout_roach1.csv')
    layout_file = os.path.join(os.getcwd(), '..', 'detector_layouts', 'roach1_pass23_shifts.npy')

    # Load KID Shifts
    shifts: dict[int, tuple[float, float]] = load_kid_layout(layout_file)

    # Load KID TODs
    kid_tods: dict[int, np.ndarray] = np.load(os.path.join(os.getcwd(), 'r1p3_norm_dfs.npy'), allow_pickle=True).item()

    downsampled_tods = {kid:downsample(tod, 200, allow_truncate=True)
                        for kid, tod in kid_tods.items()}

    # Only map KIDs both in layout and TODs
    common_kids = set(kid_tods.keys()).intersection(set(shifts.keys()))
    shifts_common = {key:val for key, val in shifts.items() if key in common_kids}
    kid_tods_common = {key:val for key, val in downsampled_tods.items() if key in common_kids}

    # Create an animated scatter plot window which shows
    # each KID's DF as its colour which changes in time
    root = tk.Tk()
    root.title("Kid Viewer")

    app = AnimatedScatterPlot(root, shifts_common, kid_tods_common, tick_ms=1, speed_mult=1)

    # Start the GUI loop
    app.mainloop()
