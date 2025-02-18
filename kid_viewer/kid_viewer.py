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
import tkinter.ttk as ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scatter_animation import AnimatedScatterPlot
import helpers as hp

# try:
#     import ttkthemes
# except ImportError:
#     print('Did not find module ttkthemes')
ttkthemes = None


# ============================================================================ #
# ENTRY POINT
# ============================================================================ #

if __name__ == '__main__':

    # ============================================================================ #
    # LOAD & PROCESS DATA
    # ============================================================================ #

    # Get kid shifts & kid tods as dictionaries
    # kid_shifts: dict[str, tuple[float, float]] = hp.get_250um_shifts()
    # kid_tods: dict[str, np.ndarray] = hp.get_250um_tods()
    kid_shifts: dict[str, tuple[float, float]] = hp.get_single_roach_shifts(roach=1, source_shifts=True)
    kid_tods: dict[str, np.ndarray] = hp.get_single_roach_tods(roach=1, downsampled=True)
    x_um, y_um = hp.get_source_offset_tods()
    x_um = hp.downsample(x_um, 200, allow_truncate=True)
    y_um = hp.downsample(y_um, 200, allow_truncate=True)

    # Downsample kid_tods by `factor`. Note: kid_tods may already be downsampled from native 477Hz
    down_sampled_tods = {kid: hp.downsample(tod, 20, allow_truncate=True) for kid, tod in kid_tods.items()}

    # Only map KIDs both in layout and TODs
    r1kids = np.load('data/roach1kids.npy', allow_pickle=True).item()
    common_kids = set(kid_tods.keys()).intersection(set(kid_shifts.keys())).intersection(r1kids)
    shifts_common = {key:val for key, val in kid_shifts.items() if key in common_kids}
    kid_tods_common = {key:val for key, val in down_sampled_tods.items() if key in common_kids}

    # Compute common mode per roach (mean value in time)
    tod_len = next(iter(kid_tods_common.values())).size
    roach_sum_tods = {}
    roach_tod_count = {}
    for kid_id, tod in kid_tods_common.items():
        roach_str = kid_id[:6]  # e.g. 'roach1'
        if roach_str not in roach_sum_tods:
            roach_sum_tods[roach_str] = np.zeros(tod_len)
            roach_tod_count[roach_str] = 1
        else:
            roach_sum_tods[roach_str] += tod
            roach_tod_count[roach_str] += 1
    roach_common_modes = {key: val / roach_tod_count[key] for key, val in roach_sum_tods.items()}

    # common mode for all roaches
    common_mode = np.mean([val for val in roach_common_modes.values()], axis=0)

    # Subtract common mode
    ct_removed_tods = {kid:kid_tod - common_mode for kid, kid_tod in kid_tods_common.items()}


    # ============================================================================ #
    # BUILD GUI ELEMENTS
    # ============================================================================ #

    # Tkinter root widget & style setup
    if ttkthemes is not None:
        root = ttkthemes.ThemedTk(theme='arc')
    else:
        print("Using default theme since ttkthemes was not found")
        root = tk.Tk()
    s = ttk.Style()
    s.configure('TFrame', background='white')
    s.configure('TScale', background='white')
    s.configure('TLabel', background='white')
    s.configure('TButton', background='white')
    root.title("BLAST-TNG KID Viewer")

    # root children
    mainframe = ttk.Frame(root, padding="3 3 12 12")

    # mainframe children
    title = ttk.Label(mainframe, text="BLAST-TNG KID VIEWER")  # TODO: generate informative header from config
    slider = ttk.Scale(mainframe, orient='horizontal')
    kid_animation = AnimatedScatterPlot(mainframe, shifts_common, x_um, y_um, ct_removed_tods,
                                        tick_ms=0, speed_mult=1, slider=slider)
    fig = hp.common_mode_plot(common_mode)
    canvas = FigureCanvasTkAgg(fig, master=mainframe)
    button_menu = ttk.Frame(mainframe)

    # button_menu children
    play_pause_btn = ttk.Button(button_menu, text="Pause",
                                command=lambda: hp.toggle_playback(kid_animation, play_pause_btn))
    root.bind("<space>", lambda event: hp.toggle_playback(kid_animation, play_pause_btn))


    # ============================================================================ #
    # ARRANGE LAYOUT
    # ============================================================================ #

    # root children
    mainframe.grid(column=0, row=0)

    # mainframe children
    title.grid(column=0, row=0)
    kid_animation.canvas.get_tk_widget().grid(column=0, row=1)
    canvas.get_tk_widget().grid(column=0, row=2)
    slider.grid(column=0, row=3, sticky='EW', padx=(75, 33))  # padx to align slider with common-mode plot
    button_menu.grid(column=0, row=4)

    # button_menu children
    play_pause_btn.grid(column=0, row=0)

    root.mainloop()


    # ============================================================================ #
    # START GUI LOOP
    # ============================================================================ #

    kid_animation.mainloop()
