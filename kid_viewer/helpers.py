# ============================================================================ #
# helpers.py
#
# Jonah Lee
#
# Helper library for KID Viewer
# Contains functions for loading TODs from local file system,
# as well as other miscellaneous functions to clean up kid_viewer.py
# ============================================================================ #

import os
from tkinter import ttk

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scatter_animation import AnimatedScatterPlot

# ============================================================================ #
# GET DICTIONARIES: kid_shifts / kid_tods
# ============================================================================ #

def get_single_roach_tods(roach: int, downsampled=False) -> dict[str, np.ndarray]:
    data_dir = os.path.join(os.getcwd(), 'data')
    tod_file = os.path.join(data_dir, f'roach_{roach}_all', f'norm_df_dict{"_ds_10" if downsampled else ""}.npy')
    tod_dict = np.load(tod_file, allow_pickle=True).item()
    return {f'roach{roach}_' + key: val for key, val in tod_dict.items()}

def get_single_roach_shifts(roach: int, source_shifts=True) -> dict[str, tuple[float, float]]:
    if source_shifts:
        layout_file = os.path.normpath(
            os.path.join(os.getcwd(),'..', 'detector_layouts', f'roach{roach}_all_shifts.npy')
        )
    else:
        layout_file = os.path.normpath(
            os.path.join(os.getcwd(), '..', 'detector_layouts', f'layout_roach{roach}.csv')
        )
    shifts_dict = load_kid_layout(layout_file)
    return {f'roach{roach}_' + key: val for key, val in shifts_dict.items()}

def get_250um_tods() -> dict[str, np.ndarray]:
    # Load KID TODs
    tods2: dict[str, np.ndarray] = get_single_roach_tods(roach=2, downsampled=True)
    tods4: dict[str, np.ndarray] = get_single_roach_tods(roach=4, downsampled=True)
    tods5: dict[str, np.ndarray] = get_single_roach_tods(roach=5, downsampled=True)
    return {**tods2, **tods4, **tods5}

def get_250um_shifts() -> dict[str, tuple[float, float]]:

    # combined_shifts = np.load(
    #     os.path.join(os.getcwd(), 'data', 'roach_245_pass_3', 'shifts_source_no_rejects.npy'),
    #     allow_pickle=True
    # ).item()

    # Load KID Shifts
    shifts2: dict[str, tuple[float, float]] = get_single_roach_shifts(roach=2, source_shifts=True)
    shifts4: dict[str, tuple[float, float]] = get_single_roach_shifts(roach=4, source_shifts=True)
    shifts5: dict[str, tuple[float, float]] = get_single_roach_shifts(roach=5, source_shifts=True)
    return {**shifts2, **shifts4, **shifts5}


# ============================================================================ #
# DATA LOADING
# ============================================================================ #

def load_kid_layout(file, rejects_file=None):
    if file.endswith('.csv'):
        return load_kid_layout_csv(file, rejects_file)
    if file.endswith('.npy'):
        return load_kid_layout_npy(file, rejects_file)

    raise ValueError('Layout file must end with .csv or .npy')

def load_kid_layout_npy(file, rejects_file=None) -> dict[str, tuple[float, float]]:
    """Loads KID x/y coordinates on the image plane for a ROACH

    Returns a dictionary which maps KID IDs to coordinate pairs.
    """
    try:
        layouts_dict = np.load(file, allow_pickle=True).item()

        # parse keys in the form '0000' or 'roach1_0000'
        if isinstance(next(iter(layouts_dict.keys())), str):
            layouts_dict = {key[-4:]: val for key, val in layouts_dict.items()}
        # parse keys in the form of ints
        if isinstance(next(iter(layouts_dict.keys())), int):
            layouts_dict = {f'{key:04}': val for key, val in layouts_dict.items()}

        if rejects_file is not None:
            try:
                rejects = np.loadtxt(rejects_file, delimiter=' ', dtype=str)
                if isinstance(rejects[0], str): rejects = [key[-4:] for key in rejects]
                if isinstance(rejects[0], int): rejects = [f'{key:04}' for key in rejects]
                layouts_dict = {key: val for key, val in layouts_dict.items() if key in rejects}
            except FileNotFoundError as err:
                print(f"File {rejects_file} not found")
                raise err

        return layouts_dict

    except FileNotFoundError as err:
        print(f"File {file} not found")
        raise err

def load_kid_layout_csv(file, rejects_file=None) -> dict[str, tuple[float, float]]:
    """Loads KID x/y coordinates on the image plane for a ROACH

    Returns a dictionary which maps KID IDs to coordinate pairs.
    """
    try:
        df = pd.read_csv(file, index_col=0)

        if rejects_file is not None:
            try:
                rejects = np.loadtxt(rejects_file, delimiter=' ', dtype=str),
                df = df[~df.index.isin(rejects)]
            except FileNotFoundError as err:
                print(f"File {rejects_file} not found")
                raise err

        return {str(kid): (df['x'][kid], df['y'][kid]) for kid in df.index}

    except FileNotFoundError as err:
        print(f"File {file} not found")
        raise err


# ============================================================================ #
# MISCELLANEOUS
# ============================================================================ #

def downsample(arr: np.ndarray, factor, allow_truncate=False):
    assert arr.ndim == 1, "can only down-sample 1-d array"
    if allow_truncate: arr = arr[:-(arr.size % factor)]
    else: assert arr.size % factor == 0, "array length must be a multiple of down-sampling factor"
    reshaped = np.reshape(arr, (-1, factor))
    return reshaped.mean(axis=1)

def toggle_playback(animation: AnimatedScatterPlot, button: ttk.Button):
    if animation.is_playing:
        animation.pause()
        button.config(text="Play")
    else:
        animation.resume()
        button.config(text="Pause")

def common_mode_plot(common_mode: np.ndarray) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.plot(common_mode)
    ax.margins(0)
    for spine in ax.spines.values(): spine.set(linewidth=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig
