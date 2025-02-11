import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from map_maker_iterative.mmi_roach import Roach
from map_maker_iterative.mmi_config import ScanPass, RoachID


def load_kid_shifts(roach) -> dict[int, tuple[float, float]]:
    """Loads KID x/y coordinates on the image plane for a ROACH

    Returns a dictionary which maps KID IDs to coordinate pairs.
    """
    pass

def load_kid_tods(roach) -> dict[int, np.ndarray]:
    """Loads the TODs (Time Ordered Data) for each KID in this ROACH

    Returns a dictionary which maps KID IDs to numpy TOD arrays.
    """
    pass

def plot_animation(shifts, tods):
    """Creates an animated scatter plot window which shows each KID's DF as its colour which changes in time"""
    pass



if __name__ == '__main__':
    # 1. Select Roach / Slice
    roach = Roach(RoachID(1), ScanPass.PASS_1)

    # 2. Load KID Shifts
    shifts = load_kid_shifts()

    # 3. Load KID TODs
    kid_tods: np.ndarray = load_kid_tods(roach)

    # 4. Animate Over Time
    plot_animation(shifts, kid_tods)