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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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


def downsample(arr: np.ndarray, factor, allow_truncate=False):
    assert arr.ndim == 1, "can only down-sample 1-d array"
    if allow_truncate: arr = arr[:-(arr.size % factor)]
    else: assert arr.size % factor == 0, "array length must be a multiple of down-sampling factor"
    reshaped = np.reshape(arr, (-1, factor))
    return reshaped.mean(axis=1)


def get_file_names(roach: int, downsampled=False, source_shifts=True):
    """Get (layout_file, tod_file) for RCW-92 observation, all passes"""
    data_dir = os.path.join(os.getcwd(), 'data')
    if source_shifts:
        layout_file = os.path.normpath(
            os.path.join(os.getcwd(),'..', 'detector_layouts', f'roach{roach}_all_shifts.npy')
        )
    else:
        layout_file = os.path.normpath(
            os.path.join(os.getcwd(), '..', 'detector_layouts', f'layout_roach{roach}.csv')
        )
    tod_file = os.path.join(data_dir, f'roach_{roach}_all', f'norm_df_dict{"_ds_10" if downsampled else ""}.npy')
    return layout_file, tod_file


def get_250um_data():
    # Define file paths
    layout_file2, tod_file2 = get_file_names(roach=2, downsampled=True, source_shifts=True)
    layout_file4, tod_file4 = get_file_names(roach=4, downsampled=True, source_shifts=True)
    layout_file5, tod_file5 = get_file_names(roach=5, downsampled=True, source_shifts=True)
    # Load KID Shifts
    shifts2: dict[str, tuple[float, float]] = load_kid_layout(layout_file2)
    shifts4: dict[str, tuple[float, float]] = load_kid_layout(layout_file4)
    shifts5: dict[str, tuple[float, float]] = load_kid_layout(layout_file5)
    # Load KID TODs
    tods2: dict[str, np.ndarray] = np.load(tod_file2, allow_pickle=True).item()
    tods4: dict[str, np.ndarray] = np.load(tod_file4, allow_pickle=True).item()
    tods5: dict[str, np.ndarray] = np.load(tod_file5, allow_pickle=True).item()
    # Make keys unique & combine
    shifts2 = {'roach2_' + key: val for key, val in shifts2.items()}
    shifts4 = {'roach4_' + key: val for key, val in shifts4.items()}
    shifts5 = {'roach5_' + key: val for key, val in shifts5.items()}
    tods2 = {'roach2_' + key: val for key, val in tods2.items()}
    tods4 = {'roach4_' + key: val for key, val in tods4.items()}
    tods5 = {'roach5_' + key: val for key, val in tods5.items()}
    combined_shifts = {**shifts2, **shifts4, **shifts5}
    # combined_shifts = np.load(
    #     os.path.join(os.getcwd(), 'data', 'roach_245_pass_3', 'shifts_source_no_rejects.npy'),
    #     allow_pickle=True
    # ).item()
    combined_tods = {**tods2, **tods4, **tods5}

    return combined_shifts, combined_tods


def toggle_playback(animation: AnimatedScatterPlot, button: ttk.Button):
    if animation.is_playing:
        animation.pause()
        button.config(text="Play")
    else:
        animation.resume()
        button.config(text="Pause")

# ============================================================================ #
# ENTRY POINT
# ============================================================================ #

if __name__ == '__main__':

    # ============================================================================ #
    # LOAD & PROCESS DATA
    # ============================================================================ #

    kid_shifts, kid_tods = get_250um_data()

    # layout_file, tod_file = get_file_names(roach=1, downsampled=True, source_shifts=True)
    # kid_shifts = load_kid_layout(layout_file)
    # kid_tods = np.load(tod_file, allow_pickle=True).item()

    down_sampled_tods = {kid:downsample(tod, 20, allow_truncate=True) for kid, tod in kid_tods.items()}

    # Only map KIDs both in layout and TODs
    common_kids = set(kid_tods.keys()).intersection(set(kid_shifts.keys()))
    shifts_common: dict[str, tuple[float, float]] = {key:val for key, val in kid_shifts.items() if key in common_kids}
    kid_tods_common: dict[str, np.ndarray] = {key:val for key, val in down_sampled_tods.items() if key in common_kids}

    tod_len = next(iter(kid_tods_common.values())).size
    common_mode = np.zeros(tod_len)
    for tod in kid_tods_common.values():
        common_mode += tod
    common_mode /= tod_len

    ct_removed_tods = {kid:kid_tod - common_mode for kid, kid_tod in kid_tods_common.items()}

    # ============================================================================ #
    # BUILD GUI ELEMENTS
    # ============================================================================ #
    root = tk.Tk()
    root.title("BLAST-TNG KID Viewer")

    # root children
    mainframe = ttk.Frame(root, padding="3 3 12 12")

    # mainframe children
    # title_text = (
    #     "ROACH 1"
    #     f"\nLayout File: {layout_file}"
    #     f"\nTOD File: {tod_file}"
    # )
    title = ttk.Label(mainframe, text="")
    slider = ttk.Scale(mainframe, orient='horizontal')
    kid_animation = AnimatedScatterPlot(mainframe, shifts_common, ct_removed_tods,
                                        tick_ms=1, speed_mult=1, slider=slider)
    button_menu = ttk.Frame(mainframe)

    # button_menu children
    play_pause_btn = ttk.Button(button_menu, text="Pause",
                                command=lambda: toggle_playback(kid_animation, play_pause_btn))

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.plot(common_mode)
    ax.margins(0)
    for spine in ax.spines.values(): spine.set(linewidth=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Create FigureCanvasTkAgg and embed the plot
    canvas = FigureCanvasTkAgg(fig, master=mainframe)
    canvas.get_tk_widget().grid(column=0, row=2)

    # ============================================================================ #
    # ARRANGE LAYOUT
    # ============================================================================ #

    # root children
    mainframe.grid(column=0, row=0)

    # mainframe children
    title.grid(column=0, row=0)
    kid_animation.canvas.get_tk_widget().grid(column=0, row=1)
    slider.grid(column=0, row=3, sticky='NWES')
    button_menu.grid(column=0, row=4)

    # button_menu children
    play_pause_btn.grid(column=0, row=0)

    root.mainloop()


    # ============================================================================ #
    # START GUI LOOP
    # ============================================================================ #

    kid_animation.mainloop()
