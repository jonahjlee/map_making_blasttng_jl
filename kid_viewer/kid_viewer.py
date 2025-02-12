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

import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from scatter_animation import AnimatedScatterPlot


# ============================================================================ #
# HELPER FUNCTIONS
# ============================================================================ #

def load_kid_layout(file, rejects_file=None):
    if file.endswith('.csv'):
        return load_kid_layout_csv(file, rejects_file)
    if file.endswith('.npy'):
        return load_kid_layout_npy(file, rejects_file)

    raise ValueError('Layout file must end with .csv or .npy')


def load_kid_layout_csv(file, rejects_file=None) -> dict[int, tuple[float, float]]:
    """Loads KID x/y coordinates on the image plane for a ROACH

    Returns a dictionary which maps KID IDs to coordinate pairs.
    """
    try:
        df = pd.read_csv(file, index_col=0)

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
        print(f"File {file} not found")
        raise err


def load_kid_layout_npy(file, rejects_file=None) -> dict[int, tuple[float, float]]:
    """Loads KID x/y coordinates on the image plane for a ROACH

    Returns a dictionary which maps KID IDs to coordinate pairs.
    """
    try:
        layouts_dict = np.load(file, allow_pickle=True).item()

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
        print(f"File {file} not found")
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

    down_sampled_tods = {kid:downsample(tod, 200, allow_truncate=True)
                         for kid, tod in kid_tods.items()}

    # Only map KIDs both in layout and TODs
    common_kids = set(kid_tods.keys()).intersection(set(shifts.keys()))
    shifts_common = {key:val for key, val in shifts.items() if key in common_kids}
    kid_tods_common = {key:val for key, val in down_sampled_tods.items() if key in common_kids}

    # Create an animated scatter plot window which shows
    # each KID's DF as its colour which changes in time
    root = tk.Tk()
    root.title("Kid Viewer")

    # ===== CREATE GUI ELEMENTS =====

    mainframe = ttk.Frame(root, padding="3 3 12 12")

    # mainframe children
    title = ttk.Label(mainframe, text="Test")
    kid_animation = AnimatedScatterPlot(mainframe, shifts_common, kid_tods_common, tick_ms=1, speed_mult=1)
    options_bar = ttk.Frame(mainframe)
    slider = ttk.Scale(mainframe, orient='horizontal')

    def toggle_playback(button):
        if kid_animation.is_playing:
            kid_animation.pause()
            button.config(text="Play")
        else:
            kid_animation.resume()
            button.config(text="Pause")

    # options_bar children
    btn1 = ttk.Button(options_bar, text="Pause", command=lambda: toggle_playback(btn1))
    btn3 = ttk.Button(options_bar, text="btn3")

    # ===== ARRANGE ELEMENTS IN WINDOW =====

    mainframe.grid(column=0, row=0)

    kid_animation.canvas.get_tk_widget().grid(column=2, row=2)
    title.grid(column=2, row=1)
    options_bar.grid(column=1, row=1, rowspan=2)
    slider.grid(column=1, row=3, columnspan=2, sticky='NWES')

    btn1.grid(column=0, row=0)
    btn3.grid(column=0, row=1)

    # Start the GUI loop
    kid_animation.mainloop()
