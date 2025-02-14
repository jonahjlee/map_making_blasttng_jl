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


# ============================================================================ #
# ENTRY POINT
# ============================================================================ #

if __name__ == '__main__':

    # ============================================================================ #
    # LOAD & PROCESS DATA
    # ============================================================================ #

    # Get kid shifts & kid tods as dictionaries
    kid_shifts: dict[str, tuple[float, float]] = hp.get_250um_shifts()
    kid_tods: dict[str, np.ndarray] = hp.get_250um_tods()

    # Downsample kid_tods by `factor`. Note: kid_tods may already be downsampled from native 477Hz
    down_sampled_tods = {kid: hp.downsample(tod, 20, allow_truncate=True) for kid, tod in kid_tods.items()}

    # Only map KIDs both in layout and TODs
    common_kids = set(kid_tods.keys()).intersection(set(kid_shifts.keys()))
    shifts_common = {key:val for key, val in kid_shifts.items() if key in common_kids}
    kid_tods_common = {key:val for key, val in down_sampled_tods.items() if key in common_kids}

    # Compute common mode (mean value in time)
    tod_len = next(iter(kid_tods_common.values())).size
    common_mode = np.zeros(tod_len)
    for tod in kid_tods_common.values():
        common_mode += tod
    common_mode /= tod_len

    # Subtract common mode
    ct_removed_tods = {kid:kid_tod - common_mode for kid, kid_tod in kid_tods_common.items()}


    # ============================================================================ #
    # BUILD GUI ELEMENTS
    # ============================================================================ #

    # Tkinter root widget
    root = tk.Tk()
    root.title("BLAST-TNG KID Viewer")

    # root children
    mainframe = ttk.Frame(root, padding="3 3 12 12")

    # mainframe children
    title = ttk.Label(mainframe, text="BLAST-TNG KID VIEWER")  # TODO: generate informative header from config
    slider = ttk.Scale(mainframe, orient='horizontal')
    kid_animation = AnimatedScatterPlot(mainframe, shifts_common, ct_removed_tods,
                                        tick_ms=1, speed_mult=1, slider=slider)
    fig = hp.common_mode_plot(common_mode)
    canvas = FigureCanvasTkAgg(fig, master=mainframe)
    button_menu = ttk.Frame(mainframe)

    # button_menu children
    play_pause_btn = ttk.Button(button_menu, text="Pause",
                                command=lambda: hp.toggle_playback(kid_animation, play_pause_btn))


    # ============================================================================ #
    # ARRANGE LAYOUT
    # ============================================================================ #

    # root children
    mainframe.grid(column=0, row=0)

    # mainframe children
    title.grid(column=0, row=0)
    kid_animation.canvas.get_tk_widget().grid(column=0, row=1)
    canvas.get_tk_widget().grid(column=0, row=2)
    slider.grid(column=0, row=3, sticky='NWES')
    button_menu.grid(column=0, row=4)

    # button_menu children
    play_pause_btn.grid(column=0, row=0)

    root.mainloop()


    # ============================================================================ #
    # START GUI LOOP
    # ============================================================================ #

    kid_animation.mainloop()
