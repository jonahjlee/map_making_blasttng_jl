# ============================================================================ #
# scatter_animation.py
#
# Jonah Lee (with use of generative AI)
#
# Scatter Animation
# Tkinter widget: display a scatter plot as an animation over time.
# Allows assignment of color for each marker for each frame (timestamp)
# ============================================================================ #


# ============================================================================ #
# IMPORTS
# ============================================================================ #

import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.wcs import WCS
from astropy.time import Time
def um_to_ra_dec(x_um, y_um, platescale=None):
    return az_el_offsets_to_ra_dec(*um_to_az_el_offsets(x_um, y_um, platescale=platescale))

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
# AnimatedScatterPlot
# ============================================================================ #

class AnimatedScatterPlot(tk.Frame):
    def __init__(self,
                 parent,
                 position_offsets: dict[str, tuple[float, float]],
                 x_um: np.ndarray, y_um: np.ndarray,
                 timestreams: dict[str, np.ndarray],
                 tick_ms: int,
                 speed_mult: int=1,
                 slider: tk.Scale=None,
                 **kwargs):
        super().__init__(parent, **kwargs)

        assert set(position_offsets.keys()) == set(timestreams.keys()), \
            "positions and timestreams must have the same keys"

        self.parent: tk.Tk = parent
        self.position_offsets: dict[str, tuple[float, float]] = position_offsets
        self.timestreams: dict[str, np.ndarray] = timestreams
        self.tick_ms: int = tick_ms
        self.speed_mult: int = speed_mult

        self.slider = None
        self.slider_val = None
        self.set_slider(slider)

        self.num_points: int = len(self.position_offsets)
        self.tod_len: int = len(next(iter(self.timestreams.values())))  # all timestreams have equal length

        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.fig.subplots_adjust(left=.1, bottom=0.1, right=.9, top=.9)
        self.ax.set_xlabel('RA')
        self.ax.set_ylabel('Dec.')
        self.ax.set_title('RCW-92 Observation at 500μm\nDF Normalized to Calibration Lamp, Common-Mode Removed')
        self.ax.set_facecolor('#DDD')

        self.x_vals = lambda i: [point[0] for point in self.position_offsets.values()] - x_um[i]
        self.y_vals = lambda i: [point[1] for point in self.position_offsets.values()] - y_um[i]
        self.ra = lambda i: um_to_ra_dec(self.x_vals(i), self.y_vals(i)).icrs.ra
        self.dec = lambda i: um_to_ra_dec(self.x_vals(i), self.y_vals(i)).icrs.dec
        self.color_vals = lambda i: [timestream[i] for timestream in self.timestreams.values()]

        self.scatter, self.colorbar, self.canvas, self.ani = None, None, None, None

        self._is_playing = True
        self.frame = 0
        self.create_animated_plot()

    def create_animated_plot(self):
        self.scatter = self.ax.scatter(
            self.ra(0),
            self.dec(0),
            s=150, marker='h',
            c=self.color_vals(0), cmap="seismic",
            vmin=-0.15, vmax=0.15,

        )
        self.ax.set_xlim((229, 230.5))
        self.ax.set_ylim((-57.3, -56.2))
        self.colorbar = self.fig.colorbar(self.scatter, label='Power\n'r'$\text{DF}(t)/\text{DF}_\text{cal} - C(t)$',
                                          fraction=0.046, pad=0.04)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.draw()

        self.ani = FuncAnimation(self.fig, self.update_plot, frames=self.get_frame,
                                 interval=self.tick_ms, repeat=True, cache_frame_data=False)

    def update_plot(self, frame):
        # Calculate the fraction of progress
        fraction_done = frame * self.speed_mult / self.tod_len
        if self.slider is not None:
            self.slider_val.set(fraction_done)

        # Update the x and y values of the points
        x_vals = np.array(self.ra(frame))  # Get the x values at the current frame
        y_vals = np.array(self.dec(frame))  # Get the y values at the current frame
        self.scatter.set_offsets(np.column_stack((x_vals, y_vals)))  # Set new x and y coordinates

        # Update the color array
        self.scatter.set_array(np.array(self.color_vals(frame)))

        # Redraw the canvas to reflect changes
        self.canvas.draw()

        return [self.scatter]  # List of artists that were updated

    @property
    def is_playing(self):
        return self._is_playing

    def pause(self):
        self.ani.pause()
        self._is_playing = False

    def resume(self):
        self.ani.resume()
        self._is_playing = True

    def get_frame(self):
        while self.frame < self.tod_len:
            yield self.frame
            self.frame += self.speed_mult

    def set_slider(self, slider: tk.Scale):
        if slider is not None:
            self.slider = slider
            self.slider_val = tk.DoubleVar()  # must use variable or else set() calls self.slider_update
            self.slider.config(command=self.slider_update, variable=self.slider_val)

    def slider_update(self, value: str):
        self.frame = int(float(value) * self.tod_len)
        if not self.is_playing:
            # update the image
            # could unpause-resume to avoid calling protected method?
            self.ani._draw_frame(self.frame)
