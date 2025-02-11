import tkinter as tk

from erfa.ufunc import ecm06
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import pandas as pd

# from array_animation import ArrayAnimation
from scatter_animation import AnimatedScatterPlot
# from map_maker_iterative.mmi_roach import RoachPass
# from map_maker_iterative.mmi_config import ScanPass, RoachID


def load_kid_layout(layout_file, rejects_file=None) -> dict[int, tuple[float, float]]:
    """Loads KID x/y coordinates on the image plane for a ROACH

    Returns a dictionary which maps KID IDs to coordinate pairs.
    """
    assert layout_file.endswith('csv')

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


# def load_kid_tods(roach: RoachPass) -> dict[int, np.ndarray]:
#     """Loads the normalized DF TODs (Time Ordered Data) for each KID in this ROACH
#
#     Returns a dictionary which maps KID IDs to numpy TOD arrays.
#     """
#     kids: list[str] = roach.kids
#     return {int(kid):roach.get_norm_kid_df(kid) for kid in kids}


def plot_animation(shifts, tods, fps) -> None:
    """Creates an animated scatter plot window which shows each KID's DF as its colour which changes in time"""

    # Main Tkinter window setup
    root = tk.Tk()
    root.title("Animated Plot")

    app = AnimatedScatterPlot(root, shifts, tods, fps)

    # Start the GUI loop
    app.mainloop()



if __name__ == '__main__':

    # 0. Define Constants
    layout_file = os.path.join(os.getcwd(), '..', 'detector_layouts', 'layout_roach1.csv')

    # 1. Select RoachPass / Slice
    # roach = RoachPass(RoachID(1), ScanPass.PASS_1)

    # 2. Load KID Shifts
    shifts: dict[int, tuple[float, float]] = load_kid_layout(layout_file)

    # 3. Load KID TODs
    # kid_tods: dict[int, np.ndarray] = load_kid_tods(roach)
    kid_tods: dict[int, np.ndarray] = np.load(os.path.join(os.getcwd(), 'r1p3_norm_dfs.npy'), allow_pickle=True).item()

    # Only map KIDs both in layout and TODs
    common_kids = set(kid_tods.keys()).intersection(set(shifts.keys()))
    shifts_common = {key:val for key, val in shifts.items() if key in common_kids}
    kid_tods_common = {key:val for key, val in kid_tods.items() if key in common_kids}

    fps = 500

    # 4. Animate Over Time
    plot_animation(shifts_common, kid_tods_common, fps)