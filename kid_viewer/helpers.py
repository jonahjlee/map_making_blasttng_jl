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
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.wcs import WCS
from astropy.time import Time

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
            # os.path.join(os.getcwd(),'..', 'detector_layouts', f'roach{roach}_all_shifts.npy')
            os.path.join(os.getcwd(), 'data', f'roach_{roach}_all', 'source_xy.npy')
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

    combined_shifts = np.load(
        os.path.join(os.getcwd(), 'data', 'roach_245_pass_3', 'source_xy.npy'),
        allow_pickle=True
    ).item()

    # Load KID Shifts
    # shifts2: dict[str, tuple[float, float]] = get_single_roach_shifts(roach=2, source_shifts=True)
    # shifts4: dict[str, tuple[float, float]] = get_single_roach_shifts(roach=4, source_shifts=True)
    # shifts5: dict[str, tuple[float, float]] = get_single_roach_shifts(roach=5, source_shifts=True)
    # combined_shifts = {**shifts2, **shifts4, **shifts5}
    return combined_shifts


# ============================================================================ #
# DATA LOADING
# ============================================================================ #

def load_kid_layout(file, rejects_file=None) -> dict[str, tuple[float, float]]:
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

def get_source_offset_tods():
    data_dir = os.path.join(os.getcwd(), 'data')
    x_um =  np.load(os.path.join(data_dir, 'master_x_um.npy'), allow_pickle=True)
    y_um =  np.load(os.path.join(data_dir, 'master_y_um.npy'), allow_pickle=True)
    return x_um, y_um


# ============================================================================ #
# COORDINATE CONVERSIONS
# ============================================================================ #

# inverted from mmi_map_lib.py
def um_to_az_el_offsets(x_um, y_um, platescale=None):
    """Converts micron offsets on image plane to az/el on-sky offsets"""
    if platescale is None:
        platescale = 5.9075e-6  # deg/um = 21.267 arcsec/mm

    w = WCS(naxis=2)
    w.wcs.crpix = [0, 0]  # center of the focal plane is tangent point
    w.wcs.crval = [0., 0.]  # source is at center in offsets map
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cdelt = [platescale, platescale]

    x_deg, y_deg = w.wcs_pix2world(x_um, y_um, 0)

    return x_deg, y_deg

def az_el_offsets_to_ra_dec(x_az_tod, y_el_tod) -> SkyCoord:
    """Convert az/el offsets from RCW 92 to on-sky (RA / Dec) coordinates"""
    # telescope position during RCW 92 slice, see map-making slide 31
    rcw_92_min_lat = -77.110
    rcw_92_max_lat = -77.070
    rcw_92_min_lon = 162.20
    rcw_92_max_lon = 162.55
    rcw_92_min_alt = 36030
    rcw_92_max_alt = 36120
    rcw_92_avg_lat = rcw_92_min_lat/2. + rcw_92_max_lat/2.
    rcw_92_avg_lon = rcw_92_min_lon/2. + rcw_92_max_lon/2.
    rcw_92_avg_alt = rcw_92_min_alt/2. + rcw_92_max_alt/2.

    rcw92_coord = SkyCoord.from_name('RCW 92')

    rcw_92_avg_location = EarthLocation(lat=rcw_92_avg_lat*u.deg, lon=rcw_92_avg_lon*u.deg, height=rcw_92_avg_alt*u.m)
    # "At 6:10 PM local time on January 6 of [2020], BLAST-TNG began its first flight"
    # subtract 13h for UTC offset
    takeoff_time = Time("2020-01-06 5:10:00", scale='utc')
    rcw_approx_time_after_takeoff = 43_000  # seconds
    rcw_92_time = takeoff_time + rcw_approx_time_after_takeoff*u.s

    # obtain absolute az/el
    altaz_frame = AltAz(obstime=rcw_92_time, location=rcw_92_avg_location)
    source_altaz = rcw92_coord.transform_to(altaz_frame)

    abs_az = source_altaz.az + x_az_tod * u.deg * -1
    abs_alt = source_altaz.alt + y_el_tod * u.deg * -1

    # Assume balloon has not travelled enough over time to impact coordinate conversion
    onsky_coords = SkyCoord(alt=abs_alt, az=abs_az, obstime=rcw_92_time, frame='altaz', location=rcw_92_avg_location)
    return onsky_coords


# ============================================================================ #
# MISCELLANEOUS
# ============================================================================ #

def downsample(arr: np.ndarray, factor, allow_truncate=False) -> np.ndarray:
    assert arr.ndim == 1, "can only down-sample 1-d array"
    if allow_truncate: arr = arr[:-(arr.size % factor)]
    else: assert arr.size % factor == 0, "array length must be a multiple of down-sampling factor"
    reshaped = np.reshape(arr, (-1, factor))
    return reshaped.mean(axis=1)

def toggle_playback(animation: AnimatedScatterPlot, button: ttk.Button) -> None:
    if animation.is_playing:
        animation.pause()
        button.config(text="Play")
    else:
        animation.resume()
        button.config(text="Pause")

def common_mode_plot(common_mode: np.ndarray) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(common_mode, label='common mode')
    ax.margins(0)
    ax.grid(True)
    ax.legend()
    fig.subplots_adjust(left=0.1, right=.95, top=0.9, bottom=0.2)
    return fig
